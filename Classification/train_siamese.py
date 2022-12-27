import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from dataset import xBDImages
from model import SiameseCNN
from utils import make_plot, score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

batch_size = settings_dict['data_sup']['batch_size']
name = settings_dict['model']['name'] + '_siamese'
model_path = 'weights/' + name
train_disasters = settings_dict['data_sup']['train_disasters']
train_paths = settings_dict['data_sup']['train_paths']
assert len(train_disasters) == len(train_paths)
merge_classes = settings_dict['merge_classes']
n_epochs = settings_dict['epochs']
starting_epoch = 1
assert starting_epoch > 0


def train(epoch):
    model.train()
    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = 0
    y_true = []
    y_pred = []
    for data in train_loader:
        x = data['x'].to(device)
        y = data['y'].to(device)
        optimizer.zero_grad()
        out = model(x)
        y_pred.append(out.cpu())
        y_true.append(y.cpu())
        loss = F.nll_loss(input=out, target=y, weight=class_weights.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.update(x.shape[0])
    pbar.close()
    y_pred = torch.cat(y_pred).detach()
    y_true = torch.cat(y_true).detach()
    accuracy, f1_macro, f1_weighted, auc = score(y_true, y_pred)
    return total_loss / len(train_loader), accuracy, f1_macro, f1_weighted, auc


@torch.no_grad()
def test(dataloader):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0
    for data in dataloader:
        x = data['x'].to(device)
        y = data['y']
        out = model(x).cpu()
        y_pred.append(out)
        y_true.append(y)
        loss = F.nll_loss(input=out, target=y, weight=class_weights)
        total_loss += loss.item()
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    accuracy, f1_macro, f1_weighted, auc = score(y_true, y_pred)
    total_loss = total_loss / len(dataloader)
    return total_loss, accuracy, f1_macro, f1_weighted, auc


@torch.no_grad()
def save_results(hold: bool=False):
    make_plot(train_loss, test_loss, 'loss', name)
    make_plot(train_acc, test_acc, 'accuracy', name)
    make_plot(train_f1_macro, test_f1_macro, 'macro_f1', name)
    make_plot(train_f1_weighted, test_f1_weighted, 'weighted_f1', name)
    make_plot(train_auc, test_auc, 'auc', name)
    np.save('results/'+name+'_loss_train.npy', train_loss)
    np.save('results/'+name+'_loss_test.npy', test_loss)
    np.save('results/'+name+'_acc_train.npy', train_acc)
    np.save('results/'+name+'_acc_test.npy', test_acc)
    np.save('results/'+name+'_macro_f1_train.npy', train_f1_macro)
    np.save('results/'+name+'_macro_f1_test.npy', test_f1_macro)
    np.save('results/'+name+'_weighted_f1_train.npy', train_f1_weighted)
    np.save('results/'+name+'_weighted_f1_test.npy', test_f1_weighted)
    np.save('results/'+name+'_auc_train.npy', train_auc)
    np.save('results/'+name+'_auc_test.npy', test_auc)
    if hold:

        hold_dataset = xBDImages(
            ['datasets/xbd/hold_bldgs/'],
            ['socal-fire'],
            merge_classes
        )
        hold_loader = DataLoader(hold_dataset, batch_size)
        hold_scores = test(hold_loader)

        model.load_state_dict(torch.load(model_path+'_best.pt'))
        hold_scores = test(hold_loader)



if __name__ == "__main__":

    train_dataset = xBDImages(train_paths, train_disasters, merge_classes)
    test_dataset = xBDImages(['datasets/xbd/test_bldgs/'], ['socal-fire'], merge_classes)

    num_classes = train_dataset.num_classes

    if settings_dict['data_sup']['leak']:
        y_all = np.fromiter((data['y'].item() for data in test_dataset), dtype=int)
        test_idx, leak_idx = train_test_split(np.arange(y_all.shape[0]), test_size=0.1, stratify=y_all, random_state=42)
        train_leak = Subset(test_dataset, leak_idx)
        test_dataset = Subset(test_dataset, test_idx)
        train_dataset = ConcatDataset([train_dataset, train_leak])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size)

    cw_name = '_'.join(text.replace('-', '_') for text in train_disasters) + '_siamese'
    if settings_dict['data_sup']['leak']:
        cw_name = cw_name + '_leaked'
    if os.path.isfile(f'weights/class_weights_{cw_name}_{num_classes}.pt'):
        class_weights = torch.load(f'weights/class_weights_{cw_name}_{num_classes}.pt')
    else:
        y_all = np.fromiter((data['y'].item() for data in train_dataset), dtype=int)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_all), y=y_all)
        class_weights = torch.Tensor(class_weights)
        torch.save(class_weights, f'weights/class_weights_{cw_name}_{num_classes}.pt')

    model = SiameseCNN(
        settings_dict['model']['hidden_units'],
        num_classes,
        settings_dict['model']['dropout_rate']
    )
    if starting_epoch > 1:
        model.load_state_dict(torch.load(model_path+'_last.pt'))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])

    best_test_auc = best_epoch = 0

    if starting_epoch == 1:
        train_loss = np.empty(n_epochs)
        test_loss = np.empty(n_epochs)
        train_acc = np.empty(n_epochs)
        test_acc = np.empty(n_epochs)
        train_f1_macro = np.empty(n_epochs)
        test_f1_macro = np.empty(n_epochs)
        train_f1_weighted = np.empty(n_epochs)
        test_f1_weighted = np.empty(n_epochs)
        train_auc = np.empty(n_epochs)
        test_auc = np.empty(n_epochs)
    else:
        train_loss = np.load('results/'+name+'_loss_train.npy')
        test_loss = np.load('results/'+name+'_loss_test.npy')
        train_acc = np.load('results/'+name+'_acc_train.npy')
        test_acc = np.load('results/'+name+'_acc_test.npy')
        train_f1_macro = np.load('results/'+name+'_macro_f1_train.npy')
        test_f1_macro = np.load('results/'+name+'_macro_f1_test.npy')
        train_f1_weighted = np.load('results/'+name+'_weighted_f1_train.npy')
        test_f1_weighted = np.load('results/'+name+'_weighted_f1_test.npy')
        train_auc = np.load('results/'+name+'_auc_train.npy')
        test_auc = np.load('results/'+name+'_auc_test.npy')

    for epoch in range(starting_epoch, n_epochs+1):
        
        train_loss[epoch-1], train_acc[epoch-1], train_f1_macro[epoch-1],\
            train_f1_weighted[epoch-1], train_auc[epoch-1] = train(epoch)
    
        torch.save(model.state_dict(), model_path+'_last.pt')

        test_loss[epoch-1], test_acc[epoch-1], test_f1_macro[epoch-1],\
            test_f1_weighted[epoch-1], test_auc[epoch-1] = test(test_loader)

        if test_auc[epoch-1] > best_test_auc:
            best_test_auc = test_auc[epoch-1]
            best_epoch = epoch
            torch.save(model.state_dict(), model_path+'_best.pt')
        
        save_results()
    
    save_results(hold=True)
