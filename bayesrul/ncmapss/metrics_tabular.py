from pathlib import Path
import os

import numpy as np
import pandas as pd
from torch.nn.functional import gaussian_nll_loss

import torch

from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.utils.metrics import (
    PICP, MPIW, p_alphalamba, rms_calibration_error, nasa_scoring_function
)
from bayesrul.utils.plotting import ResultSaver


COLUMNS = ["RMSE-", "NLL-", "RMSCE-", "MPIW", "PICP"]


def get_all_metrics(names, bayesian=True):

    if isinstance(names, str):
        names = [names]

    df = pd.DataFrame([], columns=COLUMNS)

    for name in names:
        try:
            if bayesian:
                p = Path("results/ncmapss/bayesian", name)
            else:
                p = Path("results/ncmapss/frequentist", name)
            sav = ResultSaver(p)
            results = sav.load()

            y_true = torch.tensor(results['labels'].values, device=torch.device('cuda:0'))
            y_pred = torch.tensor(results['preds'].values, device=torch.device('cuda:0'))
            std = torch.tensor(results['stds'].values, device=torch.device('cuda:0'))
            
            mpiw = MPIW(std, y_true, normalized=True).cpu().item()
            picp = PICP(y_true, y_pred, std).cpu().item()
            rmsce = rms_calibration_error(y_pred, std, y_true).cpu().item()
            nll = gaussian_nll_loss(
                y_pred, y_true, std
            ).cpu().item()

            #nasa = nasa_scoring_function(y_true, y_pred)
            rmse = torch.sqrt(((y_true - y_pred)**2).mean()).cpu().item()

            row = pd.Series(
                data = [rmse, nll, rmsce, mpiw, picp],
                index = COLUMNS,
                name=name
            )
            df.loc[name] = row
            #print(f"NASA : {nasa}")
        except FileNotFoundError:
            pass

    return df


def get_dirs_startingby(substr, bayesian=True):
    if bayesian:
        p = "results/ncmapss/bayesian"
    else:
        p = "results/ncmapss/frequentist"

    ls = os.listdir(p)
    filtered = filter(lambda x: x[:len(substr)] == substr, ls)
    
    return list(filtered)


def fuse_by_category(cats, return_all=False):
    df_means = pd.DataFrame([], columns=COLUMNS)
    df_stds = pd.DataFrame([], columns=COLUMNS)

    for cat in cats:
        names = get_dirs_startingby(cat)

        df = get_all_metrics(names)
        
        df_mean = pd.Series(df.mean(axis=0), name=cat)
        df_std = pd.Series(df.std(axis=0), name=cat)

        if (len(df_means) == 0) & (len(df_stds) == 0):
            df_means = df_mean
            df_stds = df_std
        elif (len(df_means) > 0) & (len(df_stds) > 0):
            df_means = pd.concat([df_means, df_mean], axis=1)
            df_stds = pd.concat([df_stds, df_std], axis=1)
        else:
            raise RuntimeError("Something unexpected happened.")
            
    df_means = df_means.transpose()
    df_stds = df_stds.transpose()
    
    return df_means, df_stds


if __name__ == "__main__":
    #df = get_all_metrics(["FLIPOUT", "LRT_nopretrain", 'RADIAL'])
    #print(df.to_latex())
    fuse_by_category(['LRT', 'FLIPOUT', 'RADIAL'])