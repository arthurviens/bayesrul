import numpy as np
import pandas as pd 
from bayesrul.ncmapss.dataset import NCMAPSSDataModule


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
        
    return df.drop(columns=['index'])


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