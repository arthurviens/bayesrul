import logging
import warnings
from typing import List, Any
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler

ncmapss_files = ["N-CMAPSS_DS01-005",
    "N-CMAPSS_DS02-006",
    "N-CMAPSS_DS03-012",
    "N-CMAPSS_DS04",
    "N-CMAPSS_DS05",
    "N-CMAPSS_DS06",
    "N-CMAPSS_DS07",
    "N-CMAPSS_DS08a-009",
    "N-CMAPSS_DS08c-008",
    "N-CMAPSS_DS08d-010"
]


def generate_parquet(args) -> None:
    """ Generates parquet files in args.out_path

    Parameters
    ----------
    arg : SimpleNamespace
        arguments to forward (out_path, normalization, validation...)

    Returns
    -------
    None
    """
    for filename in args.files:
        logging.info("**** %s ****" % filename)
        logging.info("normalization = " + args.normalization)
        logging.info("validation = " + str(args.validation))

        filepath = os.path.join(args.out_path, filename)

        print("Extracting dataframes...")
        df_train, df_val, df_test = extract_validation(
            filepath=filepath,
            validation=args.validation,
        )

        # Normalization
        scaler = normalize_ncmapss(df_train, arg=args.normalization)
        _ = normalize_ncmapss(df_val, scaler=scaler)
        _ = normalize_ncmapss(df_test, scaler=scaler)

        print("Generating parquet files...")
        path = Path(args.out_path, "parquet")
        path.mkdir(exist_ok=True)
        for df, prefix in zip([df_train, df_val, df_test], ["train", "val", "test"]):
            if isinstance(df, pd.DataFrame):
                df.to_parquet(f"{path}/{prefix}_{filename}.parquet")



def normalize_ncmapss(df: pd.DataFrame, arg="", scaler=None) -> Any:
    """ Normalizes a DataFrame IN PLACE. Provide arg or already fitted scaler

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.
    arg : str, optional
        Which normalizer to use. Either 'minmax' or 'standard'
    scaler: sklearn.preprocessing scaler
        Already fitted scaler to use

    Returns
    -------
    scaler: sklearn.preprocessing scaler
        Fitted scaler for re-use.

    Raises
    ------
    AssertionError
        When df is not a pd.DataFrame
    ValueError
        When neither arg or scaler is provided
        When arg is not in ['', 'minmax', 'standard']
    """

    assert isinstance(df, pd.DataFrame), f"{type(df)} is not a DataFrame"
    nosearchfor = ["unit", "cycle", "Fc", "hs", "rul"]
    columns = df.columns[~(df.columns.str.contains('|'.join(nosearchfor)))] 
    
    if scaler is None:
        if (arg is None) or (arg == ""):
            raise ValueError("No scaler or arg provided in normalize")
        elif (arg == 'min-max') or (arg == 'minmax'):
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif arg == 'standard':
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
        else:
            raise ValueError("Arg must be in ['', 'minmax', 'standard']")

    else:
        if arg != "":
            warnings.warn("Scaler provided but 'arg' parameter not empty : arg will be ignored")
        else:
            df[columns] = scaler.transform(df[columns])

    return scaler



def choose_units_for_validation(unit_repartition, validation) -> List[int]:
    """Chooses which test unit to put in val set according to wanted val %. 

    Parameters
    ----------
    unit_repartition : pd.Series
        % of data points repartition among units.
    validation : float
        Wanted % of validation data.

    Returns
    -------
    List[int]
        chosen units IDs
    """
    assert np.abs(unit_repartition.sum() - 1) <= 1e-7, \
        "Frequencies don't add up to 1 ({})".format(unit_repartition.sum())
    assert len(unit_repartition[unit_repartition == 0]) == 0

    unit_repartition.sort_values(ascending=True, inplace=True)
    val_diff = (unit_repartition - validation).sort_values()
    below_subset = unit_repartition[unit_repartition < validation]
    
    if any(np.abs(val_diff) <= 0.05): # Take the closest unit if close enough
        unit = val_diff.abs().argmin()
        units = [val_diff.index[unit]]

    elif len(below_subset) >= 2: # Possible to choose among subsets. Take the first 
        # subset with data % > validation. Choosing the best one could take 
        # 2^len(subset), that could be too much to compute
        units = []
        percent = 0
        i = 0
        while percent < validation:
            units.append(unit_repartition.index[i])
            percent += unit_repartition.values[i]
            i += 1

    else: # Take the closest unit (can't do better !)
        unit = val_diff.abs().argmin()
        units = [val_diff.index[unit]]

    assert isinstance(units, list)
    return units


def extract_validation(
    filepath, vars = ['W', 'X_s', 'X_v', 'T', 'A'], validation=0.00
):
    """Extract train, validation and test dataframe from source file.

    Parameters
    ----------
    filepath : str
        .h5 data file
    vars : str, optional
        Which variables to extract from .h5 file
    validation : float, optional
        Ratio of training samples to hold out for validation.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        Train dataframe, validation dataframe, test dataframe.
    """
    assert 0 <= validation <= 1, (
            "'validation' must be a value within [0, 1], got %.2f" % 
            validation + "."  )

    df_train, df_test = _load_data_from_file(filepath, vars=vars)
    
    # Percentage of datapoints across all units
    unit_repartition = df_train.groupby('unit')['rul'].count() \
        / df_train.groupby('unit')['rul'].count().sum()

    if validation > 0:
        units = choose_units_for_validation(unit_repartition, validation)
        df_val = df_train[df_train.unit.isin(units)]
        df_train = df_train[~(df_train.unit.isin(units))]

    else:
        df_val = pd.DataFrame([], columns=df_train.columns) # Empty
    
    return df_train, df_val, df_test


def _load_data_from_file(filepath, vars = ['W', 'X_s', 'X_v', 'T', 'A']):
    """Load data from source file into a dataframe.

    Parameters
    ----------
    file : str
        Source file.
    vars : list
        May contain 'W', 'X_s', 'X_v', 'T', 'A'
        W: Scenario Descriptors
        X_s: Measurements
        X_v: Virtual sensors
        T: Health Parameters
        A: Auxiliary Data

    Returns
    -------
    DataFrame
        Data organized into a dataframe.
    """
    assert all([x in ['W', 'X_s', 'X_v', 'T', 'A'] for x in vars]), \
        "Wrong vars provided, choose a subset of ['W', 'X_s', 'X_v', 'T', 'A']"
    assert any([x in filepath for x in ncmapss_files]), "Incorrect file name {}"\
        .format(filepath)

    if ".h5" not in filepath:
        filepath = filepath + ".h5"

    with h5py.File(filepath, 'r') as hdf:

        dev = []
        test = []
        varnames = []

        if 'W' in vars: 
            dev.append(np.array(hdf.get('W_dev')))             
            test.append(np.array(hdf.get('W_test')))
            varnames.extend(hdf.get('W_var'))
        if 'X_s' in vars:
            dev.append(np.array(hdf.get('X_s_dev')))             
            test.append(np.array(hdf.get('X_s_test')))
            varnames.extend(hdf.get('X_s_var'))
        if 'X_t' in vars:
            dev.append(np.array(hdf.get('X_t_dev')))             
            test.append(np.array(hdf.get('X_t_test')))
            varnames.extend(hdf.get('X_t_var'))
        if 'T' in vars:
            dev.append(np.array(hdf.get('T_dev')))             
            test.append(np.array(hdf.get('T_test')))
            varnames.extend(hdf.get('T_var'))
        if 'A' in vars:
            dev.append(np.array(hdf.get('A_dev')))             
            test.append(np.array(hdf.get('A_test')))
            varnames.extend(hdf.get('A_var'))
        
        # Add RUL
        dev.append(np.array(hdf.get('Y_dev')))             
        test.append(np.array(hdf.get('Y_test')))
    
    varnames = list(np.array(varnames, dtype='U20')) # Strange string types
    varnames = [str(x) for x in varnames]
    varnames.append('rul')

    dev = np.concatenate(dev, axis=1)
    test = np.concatenate(test, axis=1)

    assert (dev.shape[1] == test.shape[1]) & (test.shape[1] == len(varnames)),\
        "Dimension error in creating ncmapss. Dev {}, test {} names {}".format(
            dev.shape, test.shape, len(varnames)
        )
        
    df_train = pd.DataFrame(data = dev, columns = varnames)
    df_test = pd.DataFrame(data = test, columns = varnames)
    
    return df_train, df_test


    