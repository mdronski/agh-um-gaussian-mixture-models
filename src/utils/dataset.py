import os
from typing import List, Tuple

import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.config import DATASET_PATH


def load(dataset_name: str) -> pd.DataFrame:
    dataset_path = os.path.join(DATASET_PATH, dataset_name)
    df = pd.read_csv(dataset_path,
                     sep='[ ]+',
                     engine='python',
                     header=None)
    df.columns = df.columns.astype(str)
    df.name = dataset_name.split('.')[0]
    return df


def visualize_many(datasets: List):
    # for two column plot
    # n_col = 2
    # n_row = math.ceil(len(datasets) / n_col)
    # figsize = (n_row*10, n_col*10)
    # fig, axes = plt.subplots(n_row, n_col, sharey='none', sharex='none', figsize=figsize)
    fig, axes = plt.subplots(len(datasets), 1, sharey='none', sharex='none', figsize=(10, len(datasets)*10))
    for dataset, ax in zip(datasets, list(axes.flat)):
        if type(dataset) == tuple:
            df, cluster_res = dataset
            visualize(df, cluster_res, ax=ax)
        else:
            visualize(dataset, ax=ax)


def visualize(dataset: pd.DataFrame, clustering_results: np.ndarray = None, ax=None):
    df = dataset.copy()
    df.name = dataset.name
    if len(dataset.columns) != 2:
        df = reduce_dims(dataset)
    x = df.columns[0]
    y = df.columns[1]

    sns_arguments = {
        'data': df,
        'x': x,
        'y': y,
        'edgecolor': "none",
    }

    if clustering_results is not None:
        df['pred'] = clustering_results
        palette = sns.color_palette("Paired", len(np.unique(clustering_results)))
        sns_arguments['legend'] = False
        sns_arguments['palette'] = palette
        sns_arguments['hue'] = 'pred'

    if ax is not None:
        ax.set_xticks([])
        ax.set_yticks([])
        sns_arguments['ax'] = ax

    plot = sns.scatterplot(**sns_arguments)
    plot.set_title(df.name)
    return plot



def reduce_dims(dataset: pd.DataFrame):
    pca = PCA(n_components=2)
    dataset_2d = pca.fit_transform(dataset)
    df = pd.DataFrame(dataset_2d)
    df.columns = df.columns.astype(str)
    df.name = dataset.name
    return df
