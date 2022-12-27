import os
import os.path as osp
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric
from PIL import Image
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, Delaunay, FaceToEdge
from torchvision.transforms import ToTensor

torch.manual_seed(42)

to_tensor = ToTensor()

class xBDImages(torch.utils.data.Dataset):

    def __init__(
        self,
        paths: List[str],
        disasters: List[str],
        merge_classes: bool=False,
        transform: Callable=None) -> None:

        list_labels = []
        for disaster, path in zip(disasters, paths):
            labels = pd.read_csv(list(Path(path + disaster).glob('*.csv*'))[0], index_col=0)
            labels.drop(columns=['long','lat', 'xcoord', 'ycoord'], inplace=True)
            labels.drop(index=labels[labels['class'] == 'un-classified'].index, inplace = True)
            labels['image_path'] = path + disaster + '/'
            list_labels.append(labels)
        
        self.labels = pd.concat(list_labels)
        self.label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':3}
        self.num_classes = 3 if merge_classes else 4
        self.merge_classes = merge_classes
        self.transform = transform
    
    def __len__(self) -> int:
        return self.labels.shape[0]
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor]:

        if torch.is_tensor(idx):
            idx = idx.tolist()

        post_image_file = self.labels['image_path'][idx] + self.labels.index[idx]
        pre_image_file = post_image_file.replace('post', 'pre')
        pre_image = Image.open(pre_image_file)
        post_image = Image.open(post_image_file)
        pre_image = pre_image.resize((128, 128))
        post_image = post_image.resize((128, 128))
        pre_image = to_tensor(pre_image)
        post_image = to_tensor(post_image)
        assert pre_image.shape == post_image.shape == (3,128,128)
        images = torch.cat((pre_image, post_image),0).flatten()

        if self.transform is not None:
            images = self.transform(images)

        y = torch.tensor(self.label_dict[self.labels['class'][idx]])

        if self.merge_classes:
            y[y==3] = 2
        
        sample = {'x': images, 'y': y}

        return sample


delaunay = Compose([Delaunay(), FaceToEdge()])

class xBDMiniGraphs(torch_geometric.data.Dataset):

    def __init__(
        self,
        root: str,
        data_path: str,
        disaster_name: str,
        transform: Callable=None,
        pre_transform: Callable=None) -> None:
        
        self.path = data_path
        self.disaster = disaster_name
        self.labels = pd.read_csv(list(Path(self.path + self.disaster).glob('*.csv*'))[0], index_col=0)
        self.labels.drop(columns=['long','lat'], inplace=True)
        zone_func = lambda row: '_'.join(row.name.split('_', 2)[:2])
        self.labels['zone'] = self.labels.apply(zone_func, axis=1)
        self.zones = self.labels['zone'].value_counts()[self.labels['zone'].value_counts()>1].index.tolist()
        self.num_classes = 4

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        processed_files = []
        for zone in self.zones:
            if not ((self.labels[self.labels['zone'] == zone]['class'] == 'un-classified').all() or \
                    (self.labels[self.labels['zone'] == zone]['class'] != 'un-classified').sum() == 1):
                processed_files.append(os.path.join(self.processed_dir, f'{zone}.pt'))
        return processed_files

    def process(self) -> None:
        label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':3}
        for zone in self.zones:
            if os.path.isfile(os.path.join(self.processed_dir, f'{zone}.pt')) or \
            (self.labels[self.labels['zone'] == zone]['class'] == 'un-classified').all() or \
            (self.labels[self.labels['zone'] == zone]['class'] != 'un-classified').sum() == 1:
                continue
            print(f'Building {zone}...')
            list_pre_images = list(map(str, Path(self.path + self.disaster).glob(f'{zone}_pre_disaster*')))
            list_post_images = list(map(str, Path(self.path + self.disaster).glob(f'{zone}_post_disaster*')))
            x = []
            y = []
            coords = []

            for pre_image_file, post_image_file in zip(list_pre_images, list_post_images):
                
                annot = self.labels.loc[os.path.split(post_image_file)[1],'class']
                if annot == 'un-classified':
                    continue
                y.append(label_dict[annot])
                coords.append((self.labels.loc[os.path.split(post_image_file)[1],'xcoord'],
                                self.labels.loc[os.path.split(post_image_file)[1],'ycoord']))

                pre_image = Image.open(pre_image_file)
                post_image = Image.open(post_image_file)
                pre_image = pre_image.resize((128, 128))
                post_image = post_image.resize((128, 128))
                pre_image = to_tensor(pre_image)
                post_image = to_tensor(post_image)
                assert pre_image.shape == post_image.shape == (3,128,128)
                images = torch.cat((pre_image, post_image),0)
                x.append(images.flatten())

            x = torch.stack(x)
            y = torch.tensor(y)
            coords = torch.tensor(coords)

            data = Data(x=x, y=y, pos=coords)
            data = delaunay(data)

            edge_index = data.edge_index

            edge_attr = torch.empty((edge_index.shape[1],1))
            for i in range(edge_index.shape[1]):
                node1 = x[edge_index[0,i]]
                node2 = x[edge_index[1,i]]
                s = (torch.abs(node1 - node2)) / (torch.abs(node1) + torch.abs(node2))
                s[s.isnan()] = 1
                s = 1 - torch.sum(s)/node1.shape[0]
                edge_attr[i,0] = s.item()
            data.edge_attr = edge_attr

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'{zone}.pt'))
    
    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data
