from pathlib import Path
from bayesian_torch.layers.variational_layers import LinearReparameterization
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from bayesrul.ncmapss.dataset import NCMAPSSDataModule

import torch

class PredLogger:
    """
    Class to save and load labels and predictions on test set 
        Also saves and loads stdev for bayesian models
    """
    def __init__(self, path, filename=None):
        self.path = Path(path, 'predictions')
        self.path.mkdir(exist_ok=True)
        if filename is None:
            filename = 'preds.npy'
            
        self.file_path = Path(self.path, filename)

    def save(self, test_preds):
        if 'stds' in test_preds.keys():
            to_save = np.array([test_preds['preds'], 
                test_preds['labels'], test_preds['stds']])
        else:
            to_save = np.array([test_preds['preds'], test_preds['labels']])
        
        np.save(self.file_path, to_save)

    def load(self):
        outputs = np.load(self.file_path)
        if outputs.shape[0] == 3:
            return {'preds': outputs[0, :], 'labels': outputs[1, :],
                    'stds': outputs[2, :]}
        else:
            return {'preds': outputs[0, :], 'labels': outputs[1, :]}


def plot_rul_pred(out, std=False):
    preds = out['preds']
    labels = out['labels']
    if std:
        stds = out['stds']

    n = len(preds)
    assert n == len(labels), "Inconsistent sizes predictions {}, labels {}"\
        .format(n, len(labels))

    fig, ax = plt.subplots(figsize = (15, 5))
    ax.plot(range(n), preds, color='blue', label='predicted')
    ax.plot(range(n), labels, color='red', label='actual')
    if std:
        ax.fill_between(range(n), preds-stds, preds+stds,
        color = 'lightskyblue')
    ax.set_title("RUL prediction")
    ax.set_xlabel("Timeline")
    ax.set_ylabel("RUL")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.grid()
    plt.legend()

    return fig, ax



def findNewRul(arr):
    maxrul = np.inf
    indexes = [0]
    for i, val in enumerate(arr):
        if val > maxrul:
            indexes.append(i)
            maxrul = val 
        else:
            maxrul = val
    return indexes

def plot_one_rul_pred(out, idx, std=False):
    preds = out['preds']
    labels = out['labels']
    if std:
        stds = out['stds']

    indexes = findNewRul(labels)
    assert (idx > 0) & (idx < len(indexes)), f"Could not find {idx}th engine."
    preds = preds[indexes[idx-1]:indexes[idx]]
    labels = labels[indexes[idx-1]:indexes[idx]]
    if std:
        stds = stds[indexes[idx-1]:indexes[idx]]

    data = NCMAPSSDataModule('../data/ncmapss/', 10000, all_dset=True)
    loader = data.test_dataloader()
    xx, yy, zz = pd.Series(dtype='object'), pd.Series(dtype='object'), pd.Series(dtype='object')
    for x, y, z, t, u, v in loader:
        xx = pd.concat([xx, pd.Series(x.detach().flatten())])
        yy = pd.concat([yy, pd.Series(y.detach().flatten())])
        zz = pd.concat([zz, pd.Series(z.detach().flatten())])

    df = pd.concat([xx, yy, zz], axis=1).rename(columns={0: 'ds_id', 1: 'unit_id', 2: 'win_id'}).reset_index(drop=True)
    df = df[indexes[idx-1]:indexes[idx]].reset_index(drop=True)
    engines = df['unit_id'].unique()
    assert len(engines) == 1, f"Different engines in selected set {engines}"
    
    total_sampling_coef = 10 * (30 + (len(df.index) - 1) * 10)
    flight_hours = (df.index / df.index.max()) * total_sampling_coef / (60*60)
    if engines[0] == 1:
        flight_hours *= 3

    n = len(preds)
    assert n == len(labels), "Inconsistent sizes predictions {}, labels {}"\
        .format(n, len(labels))


    fig, ax = plt.subplots(figsize = (20, 5))
    ax.plot(flight_hours, preds, color='blue', linewidth = 0.5,label='Mean RUL predicted')
    ax.plot(flight_hours, labels, color='black', linewidth = 3, label='True RUL')
    if std:
        ax.fill_between(flight_hours, preds-1.96*stds, preds+1.96*stds, 
                color='lightskyblue', alpha=0.3, label="95% confidence")
        ax.fill_between(flight_hours, preds-1.29*stds, preds+1.29*stds, 
                color='dodgerblue', alpha=0.3, label="80% confidence")

    ax.set_title(f"RUL prediction for engine #{engines[0]:03d}")
    ax.set_xlabel("Hours of flight (accelerated deterioration)")
    ax.set_ylabel("Remaining life in flight cycles")
    #plt.ylim(-1, 100)
    ax.grid()
    plt.legend()

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    #ax.spines["left"].set_visible(False)
    #ax.spines["bottom"].set_visible(False)
    #ax.spines["bottom"].set(linewidth=2, position=['data',0])

    return fig, ax



def plot_param_distribution(dnn):
    w = np.array([])
    b = np.array([])
    for layer in dnn.modules():
        try:
            w = np.append(w, layer.state_dict()['weight'])
            b = np.append(b, layer.state_dict()['bias'])
        except KeyError as e:
            pass

    s = 3
    w_mu, w_sigma = w.mean(), w.std()
    b_mu, b_sigma = b.mean(), b.std()
    w = w[w <= w_mu + s*w_sigma]; w = w[w >= w_mu - s*w_sigma]
    b = b[b <= b_mu + s*b_sigma]; b = b[b >= b_mu - s*b_sigma]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.kdeplot(w, ax=axes[0])
    ymin, ymax = axes[0].get_ylim()
    axes[0].vlines(w.min(), ymin, ymax, colors='black')
    axes[0].vlines(w.max(), ymin, ymax, colors='black')
    sns.kdeplot(b, ax=axes[1])
    axes[0].title.set_text(f"Weights distribution $\mu$ = {round(w_mu, 5)}, $\sigma$ = {round(w_sigma, 5)}")
    ymin, ymax = axes[1].get_ylim()
    axes[1].vlines(b.min(), ymin, ymax, colors='black')
    axes[1].vlines(b.max(), ymin, ymax, colors='black')
    axes[1].title.set_text(f"Biases distribution $\mu$ = {round(b_mu, 5)}, $\sigma$ = {round(b_sigma, 5)}")
    


def get_mus_rhos(m, mu, rho):
    if isinstance(m, LinearReparameterization):
        mu.extend(m.mu_weight.flatten().detach().cpu().numpy().tolist())
        rho.extend(torch.log1p(torch.exp(m.rho_weight))\
            .flatten().detach().cpu().numpy().tolist())