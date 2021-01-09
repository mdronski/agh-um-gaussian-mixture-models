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
    df.name = dataset_name[:-4]
    return df


def visualize_many(datasets: List[List[Tuple[str, pd.DataFrame, np.ndarray]]]) -> None:
    n_row = len(datasets)
    n_col = max(len(row) for row in datasets)
    figsize = (n_col*5, n_row*5)
    _, axes = plt.subplots(n_row, n_col, sharey='none', sharex='none', figsize=figsize)
    for r in range(n_row):
        for c in range(n_col):
            ax = axes[r][c]
            title, df, cluster_res = datasets[r][c]
            visualize(df, cluster_res, ax=ax, title=title)


def visualize(dataset: pd.DataFrame, clustering_results: np.ndarray = None, ax=None, title=None) -> None:
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
    plot.set_title(title or df.name)


def reduce_dims(dataset: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=2)
    dataset_2d = pca.fit_transform(dataset)
    df = pd.DataFrame(dataset_2d)
    df.columns = df.columns.astype(str)
    df.name = dataset.name
    return df
