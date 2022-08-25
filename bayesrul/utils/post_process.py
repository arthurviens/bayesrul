import numpy as np
import pandas as pd 
from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.utils.metrics import normal_cdf

import statsmodels.api as sm 

from typing import List

from math import factorial

import torch

def findNewRul(arr: np.array) -> np.array:
    """ Finds the indexes to separate the different engines in test set.
    As test set is not shuffled, RUL values for each engine is monotonically 
    decreasing. If it jumps from 0 to x > 0, it means it is a new engine.
    This function finds the indexes where these jumps occur
    
    Parameters
    ----------
    arg : List or np.array or pd.Series...
        Iterable of RUL values  

    Returns : np.array[int]
        Found indexes
    
    Raises
    -------
    None
    """
    maxrul = np.inf
    indexes = [0]
    for i, val in enumerate(arr):
        if val > maxrul:
            indexes.append(i)
            maxrul = val 
        else:
            maxrul = val
    return indexes


def addFlightHours(df:pd.DataFrame, skip_obs:int, win_size:int ,win_step:int):
    """ Adds flight hours measure to the test results

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of results (test preds, labels...)  

    skip_obs : int
        Factor of downsampling in the dataset creation (1Hz to 0.1Hz -> 10)

    win_size : int
        Window size in the dataset creation

    win_step : int
        Steps of the window during dataset creation

    Returns : pd.DataFrame
        df with an extra column
    
    Raises
    -------
    AssertionError:
        When 'engine_id' columns is missing
    """

    assert 'engine_id' in df.columns, \
        "'engine_id' not in dataframe's columns. Use addTestFlightInfo func"

    df.reset_index(drop=False, inplace=True)
    
    # Objective : Retrieve the time in seconds for each window.
    # However, values were downsampled, windowed and windows are strided
    # Let x be the size of our dataset here. According to 
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html formula
    # x = ((total_size/skip_factor) - (win_size-1)-1) / win_step + 1
    # (Even though there is no pytorch here, window sizes work the same as for convolutions)
    # To retrieve original size (which we know was 1 Hz) we just have to inverse
    # the above expression for total_size
    # Which we do in the following, accounting for index values
    df['flight_hours'] = (
        df.groupby(["engine_id"])['index']
        .transform(
            lambda x: (
                ((x - x.min())          # Start at index 0 by engine   
                / (x.max() - x.min()))  # Divide by number of steps (0 -> 1)
                * skip_obs              # Multiplied by a factor (total size)
                * (win_size + x.count() - 1)
                * win_step              
                / (60 * 60)             # To have hours and not seconds
                )
            )
        )
        
    df.drop(columns=['index'], inplace=True)
    df['relative_time'] = df.groupby(['ds_id', 'engine_id'])['flight_hours']\
                            .transform(lambda x: x / x.max())

    return df


def addTestFlightInfo(df: pd.DataFrame, path:str = 'data/ncmapss/'):
    """ Adds flight info to the test results

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of results (test preds, labels...)  

    path : str
        where to find the dataset lmdbs to retrieve test set

    Returns : pd.DataFrame
        df with a extra columns (ds_id, traj_id, win_id and engine_id)
    
    Raises
    -------
    AssertionError:
        When column adds fail (NaNs were introduced)
    """
    data = NCMAPSSDataModule(path, 5000, all_dset=True)
    loader = data.test_dataloader()

    ds_id = pd.Series(dtype='object')  
    traj_id = pd.Series(dtype='object') 
    win_id = pd.Series(dtype='object') 
    
    for ds, traj, win, stgs, r, spl in loader:
        ds_id = pd.concat([ds_id, pd.Series(ds.detach().flatten())])
        traj_id = pd.concat([traj_id, pd.Series(traj.detach().flatten())])
        win_id = pd.concat([win_id, pd.Series(win.detach().flatten())])
    
    added_info = pd.concat([ds_id, traj_id, win_id], axis=1)\
        .rename(columns={0: 'ds_id', 1: 'unit_id', 2: 'win_id'})\
        .reset_index(drop=True)
    
    df = pd.concat([df, added_info], axis=1)

    idxs = findNewRul(df["labels"].values); idxs.extend([len(df)])
    engine_id = []
    for i in range(1,len(idxs)):
        engine_id.extend([i-1] * (idxs[i] - idxs[i-1]))
    df['engine_id'] = engine_id

    assert df.isna().sum().sum() == 0, "NaNs introduced when adding test data."

    return df



def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')



def smooth_some_columns(
    df: pd.DataFrame, 
    cols, 
    bandwidth=0.01,
) -> pd.DataFrame:
    lowess = sm.nonparametric.lowess
    if isinstance(bandwidth, int):
        bandwidths = [bandwidth] * len(cols)
    elif isinstance(bandwidth, list) & (len(bandwidth) == len(cols)):
        bandwidths = bandwidth
    else:
        raise RuntimeError(f"'Bandwidth' parameter must be int or list of same size as cols")

    pd.options.mode.chained_assignment = None

    for i, col in enumerate(cols):
        whole_column = np.array([])
        for eng in df['engine_id'].unique():
            # Smooth with lowess. Could use savitzky_golay
            column = lowess(df.loc[df['engine_id'] == eng, col].copy(), 
                            df.loc[df['engine_id'] == eng].index.copy(), 
                            bandwidths[i])
            whole_column = np.concatenate([whole_column, column[:, 1]], axis=None)
        df[col+'_smooth'] = whole_column
    pd.options.mode.chained_assignment = 'warn'

    return df


def get_ds_unit(engine):
    ds_id = engine['ds_id'].unique()
    unit_id = engine['unit_id'].unique()

    assert len(ds_id) == 1, "Multiple datasets found"
    assert len(unit_id) == 1, "Multiple units found"
    return ds_id[0], unit_id[0]


def post_process(df: pd.DataFrame, data_path='../data/ncmapss', sigma=1.96) -> pd.DataFrame:
    assert isinstance(sigma, int) or isinstance(sigma, float), \
        f"{sigma} has to be int or float"
    sigma = torch.tensor([sigma])
    
    df['preds_minus'] = df['preds'] - sigma * df['stds']
    df['preds_plus'] = df['preds'] + sigma * df['stds']

    df = addTestFlightInfo(df, path=data_path)
    df = addFlightHours(df, 10, 30, 10)

    return df