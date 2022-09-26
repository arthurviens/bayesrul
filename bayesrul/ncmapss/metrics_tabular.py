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
from bayesrul.utils.post_process import ResultSaver, post_process

"""
Reads the results of trainings and aggregates all of them to create a single
LaTeX table for reports.
"""

COLUMNS = ["RMSE-", "NLL-", "RMSCE-", "Sharp-"]


def get_all_data(names: List[str]) -> pd.DataFrame:
    if isinstance(names, str):
        names = [names]

    dfs = []

    for name in names:
        try:
            if bayesian_or_not(name):
                p = Path("results/ncmapss/bayesian", name)
            else:
                p = Path("results/ncmapss/frequentist", name)
            sav = ResultSaver(p)
            results = sav.load()
            
            dfs.append(results)

        except FileNotFoundError as e:
            print(f"In get_all_metrics, file not found {e}. Ignoring.")

    return dfs


def col_by_ds_unit(cats: List[str], col="stds"):
    """ Computes the mean of a column by dataset and unit, for each model
    
    Parameters
    ----------
    cats : list of str
        Names of the categories of results to process ['LRT', 'FLIPOUT']...  

    Returns : pd.DataFrame
        df with ds_id and unit_id as index, and the categories as columns
    """
    dfs = []
    for cat in cats: # Meow
        names = get_dirs_startingby(cat)
        all_family = get_all_data(names)

        # Compute the mean of all results across all runs of the category
        #df = pd.Panel(all_family).mean(axis=0) # Deprecated...
        df = pd.concat(all_family).reset_index().groupby('index').mean()
        
        df = post_process(df, data_path='data/ncmapss')

        df = df.reset_index().set_index(['index', 'ds_id', 'unit_id'])
        df = df[[col]].rename(columns = {col: f"{cat}"})
        dfs.append(df)
    
    all_stds = pd.concat(dfs, axis=1).reset_index().drop(columns={'index'})

    by_dsunit = all_stds.groupby(["ds_id", "unit_id"]).mean()
    by_dsunit['Total'] = by_dsunit.mean(axis=1)
    by_dsunit = by_dsunit.sort_values("Total")

    return by_dsunit

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


def results_by_unit(df):
    pass


def latex_formatted(df_mean: pd.DataFrame, df_std: pd.DataFrame=None) -> str:
    """ Formats Pandas DataFrame into LaTeX table code """
    s = df_mean.style.highlight_min(subset=df_mean.columns, 
            props="textbf:--rwrap;", axis=0)
    s = s.format(precision=3)

    return s.to_latex(hrules=True).replace('_', ' ')


def weighted(cats: List[str], cols=["labels", "preds", "relative_time"]):
    """ Computes the mean of a column by dataset and unit, for each model
    
    Parameters
    ----------
    cats : list of str
        Names of the categories of results to process ['LRT', 'FLIPOUT']...  

    Returns : pd.DataFrame
        df with ds_id and unit_id as index, and the categories as columns
    """
    dfs = []
    values = []
    for cat in cats: # Meow
        names = get_dirs_startingby(cat)
        all_family = get_all_data(names)

        # Compute the mean of all results across all runs of the category
        #df = pd.Panel(all_family).mean(axis=0) # Deprecated...
        df = pd.concat(all_family).reset_index().groupby('index').mean()
        
        df = post_process(df, data_path='data/ncmapss')

        df = df.reset_index().set_index(['index', 'ds_id', 'unit_id'])
        
        df = df[cols]
        for col in cols:
            df.rename(columns = {col: f"{cat}_{col}"}, inplace=True)

        df[f"{cat}_rmse"] = np.sqrt((df[f"{cat}_labels"] - df[f"{cat}_preds"])**2)
        to_divide = df[f"{cat}_relative_time"].sum()
        df[f"{cat}_weighted"] = df[f"{cat}_rmse"] * df[f"{cat}_relative_time"]
        values.append(df[f"{cat}_weighted"].sum() / to_divide)
        
        dfs.append(df[f"{cat}_weighted"])
    
    all_weighted = pd.concat(dfs, axis=1).reset_index().drop(columns={'index'})

    values = pd.Series(data=values, index=cats)
    return values, all_weighted


if __name__ == "__main__":
    #df = col_by_ds_unit(["LRT", "FLIPOUT", "RADIAL", "MC_DROPOUT", "DEEP_ENSEMBLE", "HETERO_NN"], "stds")
    vals, _ = weighted(["LRT", "FLIPOUT", "RADIAL", "MC_DROPOUT", "DEEP_ENSEMBLE", "HETERO_NN"])
    df_means, _ = fuse_by_category(["LRT", "FLIPOUT", "RADIAL", "MC_DROPOUT", "DEEP_ENSEMBLE", "HETERO_NN"])
    df_means = pd.concat([vals, df_means], axis=1).rename(columns = {0: "RMSE_Weighted-"})

    print(latex_formatted(df_means))