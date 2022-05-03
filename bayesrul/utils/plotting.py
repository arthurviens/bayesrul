from pathlib import Path
from bayesian_torch.layers.variational_layers import LinearReparameterization
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import torch

class PredLogger:
    """
    Class to save and load labels and predictions on test set 
        Also saves and loads stdev for bayesian models
    """
    def __init__(self, path):
        self.path = Path(path, 'predictions')
        self.path.mkdir(exist_ok=True)
        self.file_path = Path(self.path, 'preds.npy')

    def save(self, test_preds):
        if hasattr(test_preds, 'std'): 
            to_save = np.array([test_preds['preds'], 
                test_preds['labels'], test_preds['std']])
        else:
            to_save = np.array([test_preds['preds'], test_preds['labels']])
        
        np.save(self.file_path, to_save)

    def load(self):
        outputs = np.load(self.file_path)
        if outputs.shape[0] == 3:
            return {'preds': outputs[0, :], 'labels': outputs[1, :],
                    'std': outputs[2, :]}
        else:
            return {'preds': outputs[0, :], 'labels': outputs[1, :]}


def plot_rul_pred(out, std=False):
    preds = out['preds']
    labels = out['labels']
    if std:
        stds = out['std']

    n = len(preds)
    assert n == len(labels), "Inconsistent sizes predictions {}, labels {}"\
        .format(n, len(labels))


    fig = plt.figure(figsize = (15, 5))
    ax = plt.gca()
    plt.plot(range(n), preds, color='blue', label='predicted')
    plt.plot(range(n), labels, color='red', label='actual')
    if std:
        plt.fill_between(range(n), preds-stds, preds+stds,
        color = 'lightskyblue')
    plt.title("RUL prediction")
    plt.xlabel("Timeline")
    plt.ylabel("RUL")
    plt.grid()
    plt.legend()

    return fig, ax


def get_mus_rhos(m, mu, rho):
    if isinstance(m, LinearReparameterization):
        mu.extend(m.mu_weight.flatten().detach().cpu().numpy().tolist())
        rho.extend(torch.log1p(torch.exp(m.rho_weight))\
            .flatten().detach().cpu().numpy().tolist())