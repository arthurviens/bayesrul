from pathlib import Path
import numpy as np
import pandas as pd 
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.utils.post_process import findNewRul

from abc import ABC, abstractmethod
from typing import List, Union, Dict

import torch



class Saver: # Deprecated
    def __init__(self, path):
        self.path = Path(path, 'predictions')
        self.path.mkdir(exist_ok=True)

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def load(self):
        ...


class ResultSaver:
    def __init__(self, path: str, filename: str = None) -> None:
        self.path = Path(path, 'predictions')
        self.path.mkdir(exist_ok=True)
        if filename is None:
            filename = 'results.parquet'
        self.file_path = Path(self.path, filename)

    def save(self, df: pd.DataFrame) -> None:
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        assert isinstance(df, pd.DataFrame), f"{type(df)} is not a dataframe"
        df.to_parquet(self.file_path)

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.file_path)

    def append(self, series: Union[List[pd.Series], Dict[str, np.array]]) -> None:
        if isinstance(series, list):
            series = pd.concat(series, axis=1)
        if isinstance(series, dict):
            series = pd.DataFrame(series)
        df = self.load()
        df = pd.concat([df, series], axis=1)
        assert isinstance(df, pd.DataFrame), f"{type(df)} is not a dataframe"
        s = df.isna().sum()
        if isinstance(s, pd.Series): s = s.sum()
        assert s == 0, "NaNs introduced in results dataframe"
        self.save(df)


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

"""
def plot_one_rul_pred(df: pd.DataFrame, idx: int):
    alt.data_transformers.disable_max_rows()

    engine = df[df.engine_id == idx]

    base = alt.Chart(engine).transform_calculate(
        true="'True RUL'",
        pred="'Predicted RUL'",
        interval="'Confidence Interval'",
    )

    scale = alt.Scale(domain=["True RUL", "Predicted RUL", "Confidence Interval"], range=['black', 'darkblue', 'lightblue'])

    labels = base.mark_line(color='red').encode(
        x = alt.X("flight_hours", title="Flight hours"),
        y = alt.Y("labels_smooth:Q", scale=alt.Scale(domain=(0, 100), clamp=True), 
                    title="RUL (in number of cycles)"),
        color = alt.Color('true:N', scale=scale, title='')
        )

    preds = base.mark_line(color='black').encode(
            x = alt.X("flight_hours", title="Flight hours"),
            y = alt.Y("preds_smooth:Q"), 
            color = alt.Color('pred:N', scale=scale, title='')
        )

    confidence = base.mark_area(opacity=0.9, color="#9ecae9").encode(
            x = alt.X("flight_hours", title="Flight hours"),
            y = alt.Y("preds_minus_smooth:Q"), 
            y2 = alt.Y2("preds_plus_smooth:Q"),
            color=alt.Color('interval:N', scale=scale, title='')
        )

    chart = alt.layer(
        confidence + preds + labels
    ).configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        orient='top-right'
    ).properties(width=800)\
        .configure_axisY(titleAngle=0, titleY=-10, titleAnchor="start")\
        .configure_axis(titleFontSize=16, labelFontSize=12)'''

    return chart
"""


def plot_uncertainty(preds, unc, idx):
    raise RuntimeError("Deprecated")
    pred_var = unc['pred_var']
    ep_var = unc['ep_var']
    al_var = unc['al_var']

    pred_unc = np.sqrt(pred_var)
    ep_unc = np.sqrt(ep_var)
    al_unc = np.sqrt(al_var)

    fig, ax = plt.subplots()
    ax.plot(range(len(preds)), pred_unc, label="Predictive uncertainty")
    ax.plot(range(len(preds)), ep_unc, label="Epistemic uncertainty")
    ax.plot(range(len(preds)), al_unc, label="Aleatoric uncertainty")


    indexes = findNewRul(labels)
    assert (idx > 0) & (idx < len(indexes)), f"Could not find {idx}th engine."
    preds = preds[indexes[idx-1]:indexes[idx]]
    labels = labels[indexes[idx-1]:indexes[idx]]

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

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize = (20, 5))
    axes[0].plot(flight_hours, preds, color='blue', linewidth = 0.5,label='Mean RUL predicted')
    axes[0].plot(flight_hours, labels, color='black', linewidth = 3, label='True RUL')
    
    axes[0].fill_between(flight_hours, preds-ep_unc, preds+ep_unc, 
            color='lightskyblue', alpha=0.3, label="Epistemic uncertainty")

    axes[0].set_title(f"RUL prediction for engine #{engines[0]:03d}")
    axes[0].set_xlabel("Hours of flight (accelerated deterioration)")
    axes[0].set_ylabel("Remaining life in flight cycles")
    #plt.ylim(-1, 100)
    axes[0].grid()
    axes[0].legend()

    axes[0].spines["right"].set_visible(False)
    axes[0].spines["top"].set_visible(False)


    axes[1].plot(flight_hours, preds, color='blue', linewidth = 0.5,label='Mean RUL predicted')
    axes[1].plot(flight_hours, labels, color='black', linewidth = 3, label='True RUL')
    
    axes[1].fill_between(flight_hours, preds-al_unc, preds+al_unc, 
            color='lightskyblue', alpha=0.3, label="Aleatoric uncertainty")

    axes[1].set_title(f"RUL prediction for engine #{engines[0]:03d}")
    axes[1].set_xlabel("Hours of flight (accelerated deterioration)")
    axes[1].set_ylabel("Remaining life in flight cycles")
    #plt.ylim(-1, 100)
    axes[1].grid()
    axes[1].legend()

    axes[1].spines["right"].set_visible(False)
    axes[1].spines["top"].set_visible(False)

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
    

