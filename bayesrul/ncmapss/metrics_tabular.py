from pathlib import Path
from typing import List, Tuple
import os
import re

import numpy as np
import pandas as pd
from torch.nn.functional import gaussian_nll_loss

import torch

from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.utils.metrics import (
    rms_calibration_error, sharpness
)
from bayesrul.utils.post_process import ResultSaver

"""
Reads the results of trainings and aggregates all of them to create a single
LaTeX table for reports.
"""

COLUMNS = ["RMSE-", "NLL-", "RMSCE-", "Sharp-"]


def get_all_metrics(names: List[str]) -> pd.DataFrame:
    """ Computes all the wanted metrics for specific result directories

    Parameters
    ----------
    names : list of str
        Names of the results to compute (LRT_000, LRT_001...)  

    Returns : pd.DataFrame
        df with a extra columns (ds_id, traj_id, win_id and engine_id)
    """

    if isinstance(names, str):
        names = [names]

    df = pd.DataFrame([], columns=COLUMNS)

    for name in names:
        try:
            if bayesian_or_not(name):
                p = Path("results/ncmapss/bayesian", name)
            else:
                p = Path("results/ncmapss/frequentist", name)
            sav = ResultSaver(p)
            results = sav.load()

            y_true = torch.tensor(results['labels'].values, device=torch.device('cuda:0'))
            y_pred = torch.tensor(results['preds'].values, device=torch.device('cuda:0'))
            std = torch.tensor(results['stds'].values, device=torch.device('cuda:0'))
            
            sharp = sharpness(std).cpu().item()
            rmsce = rms_calibration_error(y_pred, std, y_true).cpu().item()
            nll = gaussian_nll_loss(
                y_pred, y_true, std
            ).cpu().item()

            #nasa = nasa_scoring_function(y_true, y_pred)
            rmse = torch.sqrt(((y_true - y_pred)**2).mean()).cpu().item()

            row = pd.Series(
                data = [rmse, nll, rmsce, sharp],
                index = COLUMNS,
                name=name
            )
            df.loc[name] = row
            #print(f"NASA : {nasa}")
        except FileNotFoundError as e:
            print(f"In get_all_metrics, file not found {e}. Ignoring.")

    return df


def bayesian_or_not(s: str) -> bool:
    """ Is the model a bayesian model or frequentist model?

    Parameters
    ----------
    s : str
        Model Name

    Returns : bool
        True if the model is bayesian, False otherwise
    
    Raises
    -------
    ValueError:
        When model is unknown
    
    """
    
    if (re.search(r'\d+$', s)): # Removes numbers at the end (LRT_001 -> LRT)
        s = '_'.join(s.split('_')[:-1])
    if s.upper() in ["MFVI", "RADIAL", "LOWRANK", "LRT", "FLIPOUT"]:
        return True
    elif s.upper() in ["MC_DROPOUT", "DEEP_ENSEMBLE", "HETERO_NN"]:
        return False
    else:
        raise ValueError(f"Unknow model {s}. Choose from mfvi, lrt, flipout, "
            "radial, lowrank, mc_dropout, deep_ensemble, hetero_nn ")


def get_dirs_startingby(substr: str) -> List[str]:
    """ Gets all directory paths starting by a specific string
        If there exists directories LRT_000, LRT_001, LRT_002, calling this function
        with substr='LRT' will return all 3 paths

    Parameters
    ----------
    substr : str
        String to find

    Returns : List of strings
        Paths to the directories
    """
    if bayesian_or_not(substr):
        p = "results/ncmapss/bayesian"
    else:
        p = "results/ncmapss/frequentist"

    ls = os.listdir(p)
    filtered = filter(lambda x: x[:len(substr)] == substr, ls)
    
    return list(filtered)


def fuse_by_category(cats: List[str]) -> Tuple[pd.DataFrame]:
    """
    Parameters
    ----------
    cats : List of str
        Categories to fuse  

    Returns : pd.DataFrame, pd.DataFrame
        means and stds of the categories on all metrics of COLUMNS
    
    Raises
    -------
    RuntimeError
        When only one of (mean, std) is computed and not the other

    """
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


def latex_formatted(df_mean: pd.DataFrame, df_std: pd.DataFrame) -> str:
    """ Formats Pandas DataFrame into LaTeX table code """
    s = df_mean.style.highlight_min(subset=COLUMNS, 
            props="textbf:--rwrap;", axis=0)
    s = s.format(precision=3)

    return s.to_latex(hrules=True).replace('_', ' ')


if __name__ == "__main__":
    #df = get_all_metrics(["FLIPOUT", "LRT_nopretrain", 'RADIAL'])
    #print(df.to_latex())
    m, sd = fuse_by_category(['LRT', 'FLIPOUT', 'RADIAL', 'MC_DROPOUT', 
                                'DEEP_ENSEMBLE', 'HETERO_NN'])
    print(latex_formatted(m, sd))