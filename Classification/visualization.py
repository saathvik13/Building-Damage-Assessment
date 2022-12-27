from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from torch_geometric.utils import to_networkx


class CmapString:
    def __init__(self, palette: str, domain: Iterable[str]) -> None:
        self.domain = domain
        domain_unique = np.unique(domain)
        self.hash_table = {key: i_str for i_str, key in enumerate(domain_unique)}
        self.mpl_cmap = matplotlib.cm.get_cmap(palette, lut=len(domain_unique))

    def color(self, x: str, **kwargs):
        return self.mpl_cmap(self.hash_table[x], **kwargs)
    
    def color_list(self, **kwargs):
        return [self.mpl_cmap(self.hash_table[x], **kwargs) for x in self.domain]


def plot_on_map(labels: pd.DataFrame, color:str = 'class') -> None:
    if color == 'zone':
        zone = lambda row: '_'.join(row.name.split('_', 2)[:2])
        labels['zone'] = labels.apply(zone, axis=1)
        cmap = CmapString(palette='viridis', domain=labels['zone'].values)
        color = cmap.color_list()
        color_dict = {}
    elif color == 'class':
        color_dict = {
        "no-damage": 'green',
        "minor-damage": 'blue',
        "major-damage": 'darkorange',
        "destroyed": 'red',
        "un-classified": 'white'
        }
    else:
        raise ValueError("'Color' argument can either be 'zone' or 'class'.")
    fig = px.scatter_mapbox(
        data_frame=labels,
        lat='lat',
        lon='long',
        color=color,
        color_discrete_map=color_dict,
        mapbox_style='open-street-map',
        hover_name='class',
        zoom=10
    )
    fig.layout.update(showlegend=False)
    fig.show()


def plot_graph(data_path: str, image_path: str, save_fig=False):
    image = plt.imread(image_path)
    fig=plt.figure()
    fig.set_size_inches(30, 30)
    plt.imshow(image)
    data = torch.load(data_path)
    datax = to_networkx(data)
    pos = dict(enumerate(data.pos.numpy()))
    color_dict = {
        0: (0, 1, 0),
        1: (0, 0, 1),
        2: (1, 0.27, 0),
        3: (1, 0, 0)
    }
    colors = [color_dict[y] for y in data.y.numpy()]
    #pos = {node: (x,y) for (node, (x,y)) in pos.items()}
    nx.draw_networkx(datax, pos=pos, arrows=False, with_labels=False, node_size=100, node_color=colors)
    custom_circles = [Circle((0,0), radius=0.2, color=(0, 1, 0)), Circle((0,0), radius=0.2, color=(0, 0, 1)),
                      Circle((0,0), radius=0.2, color=(1, 0.27, 0)), Circle((0,0), radius=0.2, color=(1, 0, 0))]
    plt.legend(custom_circles, ['no-damage', 'minor-damage', 'major-damage', 'destroyed'], prop={'size':15})
    plt.axis('off')
    if save_fig:
        plt.savefig('graph_image.png', dpi=100)
    plt.show()


#################################################################################################
#The following functions are taken from:
#https://medium.com/analytics-vidhya/xview-2-challenge-part-3-exploring-the-dataset-ec924303b0df
#################################################################################################
import json

from PIL import Image, ImageDraw
from shapely import wkt


def read_label(label_path):
    with open(label_path) as json_file:
        image_json = json.load(json_file)
    return image_json

damage_dict = {
    "no-damage": (0, 255, 0, 50),
    "minor-damage": (0, 0, 255, 50),
    "major-damage": (255, 69, 0, 50),
    "destroyed": (255, 0, 0, 50),
    "un-classified": (255, 255, 255, 50)
}

def get_damage_type(properties):
    if 'subtype' in properties:
        return properties['subtype']
    else:
        return 'no-damage'

def annotate_img(draw, coords):
        wkt_polygons = []

        for coord in coords:
            damage = get_damage_type(coord['properties'])
            wkt_polygons.append((damage, coord['wkt']))

        polygons = []

        for damage, swkt in wkt_polygons:
            polygons.append((damage, wkt.loads(swkt)))

        for damage, polygon in polygons:
            x,y = polygon.exterior.coords.xy
            coords = list(zip(x,y))
            draw.polygon(coords, damage_dict[damage])

        del draw

def display_img(json_path: str, time: str='post', annotated: bool=True):
    if time=='pre':
        json_path = json_path.replace('post', 'pre')
        
    img_path = json_path.replace('labels', 'images').replace('json','png')
        
    image_json = read_label(json_path)
    img_name = image_json['metadata']['img_name']
        
    print(img_name)
    
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img, 'RGBA')
    
    if annotated:
        annotate_img(draw, image_json['features']['xy'])

    return img

def plot_image(label: str, save_fig: bool=False) -> None:

    # read images
    img_A = display_img(label, time='pre', annotated=False)
    img_B = display_img(label, time='post', annotated=False)
    img_C = display_img(label, time='pre', annotated=True)
    img_D = display_img(label, time='post', annotated=True)

    # display images
    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(30, 30)
    TITLE_FONT_SIZE = 24
    ax[0][0].imshow(img_A);
    ax[0][0].set_title('Pre Diaster Image (Not Annotated)', fontsize=TITLE_FONT_SIZE)
    ax[0][1].imshow(img_B);
    ax[0][1].set_title('Post Diaster Image (Not Annotated)', fontsize=TITLE_FONT_SIZE)
    ax[1][0].imshow(img_C);
    ax[1][0].set_title('Pre Diaster Image (Annotated)', fontsize=TITLE_FONT_SIZE)
    ax[1][1].imshow(img_D);
    ax[1][1].set_title('Post Diaster Image (Annotated)', fontsize=TITLE_FONT_SIZE)
    if save_fig:
        plt.savefig('split_image.png')
    plt.show()
