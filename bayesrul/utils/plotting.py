from types import SimpleNamespace
import matplotlib.pyplot as plt
from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.inference.vi_bnn import VI_BNN

import numpy as np
import pandas as pd
import seaborn as sns

import torch


"""
We were unable to encapsulate altair vizualization code inside functions,
hence visualization functions are in notebooks/plot_results.ipynb

Here, we have some functions for extracting and plotting scale densities
"""


def get_bnn_scales(
    bnn_name: str, 
    data_path: str="data/ncmapss", 
    out_path: str="results/ncmapss"
) -> np.array :
    """ Extracts scales from tyxe VariationalBNN
    
    Parameters
    ----------
    bnn_name : str
        Model Name
    data_path : str
        Where to find the data
    out_path : 
        Where to find the model

    Returns : np.array
        Scale parameters concatenated in one big numpy array
    
    Raises
    -------
    ValueError:
        When unexpected parameter appears
    """
    args = SimpleNamespace(
        data_path=data_path,
        out_path=out_path,
        model_name = bnn_name,
    )
    hyp = {}
    data = NCMAPSSDataModule(data_path, 5000, all_dset=False)

    gpus = [0,1,2] # Tries these 3 GPUS
    for i in gpus:
        try:
            module = VI_BNN(args, data, hyp, GPU=i)
            module._define_model()
            module.fit(0, another_GPU=i)
        except RuntimeError:
            continue
        break

    bnn = module.bnn.bnn

    #loc_params = np.array([])
    scale_params = np.array([])
    for name, param in bnn.net_guide._modules['layers'].named_pyro_params():
        if 'loc' in name:
            pass
    #        loc_params = np.concatenate([loc_params, param.detach().flatten().cpu().numpy()])
        elif 'scale' in name:
            scale_params = np.concatenate([scale_params, param.detach().flatten().cpu().numpy()])
        else: 
            raise ValueError(f'what is this param {name}')
    
    return scale_params


def plot_scales_distribution(bnn_names, data_path="../data/ncmapss", out_path="../results/ncmapss"):
    """ Plots the distribution of scales density for multiple models

    Parameters
    ----------
    bnn_names : list of str
        Models Names

    Returns : pd.DataFrame
        Plotted data
    """
    df = pd.DataFrame([])
    for bnn_name in bnn_names:
        scale_params = get_bnn_scales(bnn_name, data_path=data_path, out_path=out_path)  

        df[bnn_name] = scale_params

    df = df.melt(value_vars=bnn_names).rename(columns={'variable':'model'})
    
    with sns.axes_style('darkgrid'), sns.plotting_context("notebook", font_scale=0.9):
        sns.displot(
            df.melt(value_vars=['LRT_000', 'FLIPOUT_000', 'RADIAL_000'])\
                .rename(columns={'variable':'model'}), 
            x="value", 
            hue="model", 
            kind="hist",
        ).set(
            title="Histogram of values of scales for each model",
            xlabel="Scale value of NN parameters",
            ylabel="Number of values",
        )

    return df