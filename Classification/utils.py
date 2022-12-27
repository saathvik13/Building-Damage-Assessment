from typing import Tuple, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Merges the last two classes into a single class.
def merge_classes(data_torch):

    data_torch.y[data_torch.y==3] = 2
    return data_torch

# Calculate accuracy, F1 Score, Weighted F1 Score and ROC AUC Score
def score(y_actual, y_prediction) -> Tuple[float]:

    acc = accuracy_score(y_actual, y_prediction.argmax(dim=1, keepdim=True))
    f1m = f1_score(y_actual, y_prediction.argmax(dim=1, keepdim=True), average='macro')
    f1w = f1_score(y_actual, y_prediction.argmax(dim=1, keepdim=True), average='weighted')
    auc_score = roc_auc_score(y_actual, np.exp(y_prediction), average='macro', multi_class='ovr')
    
   
    return acc, f1m, f1w, auc_score


def make_plot(train: Iterable, test: Iterable, plot_type: str, model_name: str) -> None:
    plt.figure()
    plt.plot(train)
    plt.plot(test)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel(plot_type)
    plt.savefig('results/'+model_name+'_'+plot_type+'.pdf')
    plt.close()
