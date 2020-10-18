import os
import pandas as pd
from sklearn.decomposition import PCA

from src.config import DATASET_PATH


def load(dataset_name: str) -> pd.DataFrame:
    dataset_path = os.path.join(DATASET_PATH, dataset_name)
    df = pd.read_csv(dataset_path,
                     sep='[ ]+',
                     engine='python',
                     header=None)
    return df


def visualize(dataset: pd.DataFrame):
    if len(dataset.columns) == 2:
        visualize_2d(dataset)
    else:
        visualize_ndim(dataset)


def visualize_2d(dataset: pd.DataFrame):
    x = dataset.columns[0]
    y = dataset.columns[1]
    dataset.plot.scatter(x=x, y=y, marker='.')


def visualize_ndim(dataset: pd.DataFrame):
    pca = PCA(n_components=2)
    dataset_2d = pca.fit_transform(dataset)
    visualize_2d(pd.DataFrame(dataset_2d))