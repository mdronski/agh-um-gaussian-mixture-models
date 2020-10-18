import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from src.config import DATASET_PATH


def load(dataset_name: str) -> pd.DataFrame:
    dataset_path = os.path.join(DATASET_PATH, dataset_name)
    df = pd.read_csv(dataset_path,
                     sep='[ ]+',
                     engine='python',
                     header=None)
    df.columns = df.columns.astype(str)
    return df


def visualize(dataset: pd.DataFrame, clustering_results: np.ndarray = None):
    df = dataset.copy()
    if len(dataset.columns) != 2:
        df = reduce_dims(dataset)
    x = df.columns[0]
    y = df.columns[1]

    if clustering_results is not None:
        df['pred'] = clustering_results
        palette = sns.color_palette("Paired", len(np.unique(clustering_results)))
        sns.scatterplot(data=df, x=x, y=y,
                        edgecolor="none",
                        legend=False,
                        palette=palette,
                        hue='pred')
    else:
        sns.scatterplot(data=df, x=x, y=y, edgecolor="none")


def reduce_dims(dataset: pd.DataFrame):
    pca = PCA(n_components=2)
    dataset_2d = pca.fit_transform(dataset)
    df = pd.DataFrame(dataset_2d)
    df.columns = df.columns.astype(str)
    return df
