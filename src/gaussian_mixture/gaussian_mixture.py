import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from src.utils.dataset import visualize


def evaluate_case(dataset: pd.DataFrame, n_components=20, ax=None, draw_ellipses=False, **kwargs):
    gmm = BayesianGaussianMixture(n_components=n_components, max_iter=200, **kwargs)
    gmm.fit(dataset)
    plot = visualize(dataset, gmm.predict(dataset), ax=ax)
    if draw_ellipses:
        make_ellipses(gmm, plot)
    return plot


def make_ellipses(gmm, ax):
    colors = sns.color_palette("Paired", gmm.n_components)
    for n in range(gmm.n_components):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = Ellipse(gmm.means_[n, :2],
                      v[0], v[1],
                      180 + angle,
                      color=colors[n]
                      )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.3)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


def find_best_component_number(dataset: pd.DataFrame) -> int:
    lowest_bic = np.infty
    bic = []
    component_range = range(1, 25)
    try:
        for n_components in component_range:
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(dataset)
            bic.append(gmm.bic(dataset))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_n_component = n_components
    except Exception as e:
        pass

    # plt.xticks(component_range)
    # plt.title("BIC scores")
    # plt.bar(component_range, bic)
    # plt.show()
    return best_n_component