import wandb
import torch.nn as nn
import torch
import seaborn as sns
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import os
from sklearn import manifold
import matplotlib
import matplotlib.colors as col
import utils

matplotlib.use('Agg')

def visualize(source_feature, target1_feature,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target1_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """

    features = np.concatenate([source_feature, target1_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target1_feature))))

    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=2)
    plt.savefig(filename)
