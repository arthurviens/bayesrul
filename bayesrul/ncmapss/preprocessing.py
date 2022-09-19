import logging
import warnings
from typing import List, Any, NamedTuple, Callable, Dict, Iterator, Union
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import h5py
import os
import re

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ..utils.lmdb_utils import create_lmdb, make_slice

################################################################################
################################################################################
#
# Add preprocessing for 'A' subset : one-hot or entity embedding for flight 
# class etc.
#
################################################################################
################################################################################



ncmapss_files = [
    "N-CMAPSS_DS01-005",
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

ncmapss_data = ['X_s', 'X_v', 'T', 'A']

ncmapss_datanames = {
    'W': ['alt', 'Mach', 'TRA', 'T2'],
    'X_s': ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 
        'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf'],
    'X_v': ['T40', 'P30', 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 
        'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi'],
    'T': ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod',
        'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod',
        'LPT_eff_mod', 'LPT_flow_mod'],
    'A': [], # ['Fc', 'unit', 'cycle', 'hs'] removed because not judged relevant
    'Y': ['rul']
}


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
    if args.bits == 32:
        typ = np.float32
    else:
        typ = np.float64

    columns, mean, std = compute_scalers(args, typ)

    tr = []; te = []; vl = []
    for i, filename in enumerate(args.files):
        logging.info("**** %s ****" % filename)
        logging.info("normalization = standardization") #+ args.normalization)
        logging.info("validation = " + str(args.validation))

        filepath = os.path.join(args.out_path, filename)

        print(f"Extracting dataframes of {filename}...")

        train, val, test = extract_validation(
            filepath=filepath,
            typ=typ,
            vars=args.subdata,
            validation=args.validation,
        )
            
        match = re.search(r'DS[0-9]{2}', filename) # Extract DS0?
        
        filename = match[0]

        if args.moving_avg:
            saved_cols = train[['unit', 'cycle', 'Fc', 'rul']]
            train = train.drop(columns=['Fc', 'rul']).groupby(['unit', 'cycle']).transform(lambda x: x.rolling(10, 1).mean())
            train = pd.concat([train, saved_cols], axis=1)
            saved_cols = test[['unit', 'cycle', 'Fc', 'rul']]
            test = test.drop(columns=['Fc', 'rul']).groupby(['unit', 'cycle']).transform(lambda x: x.rolling(10, 1).mean())
            test = pd.concat([test, saved_cols], axis=1)
            saved_cols = val[['unit', 'cycle', 'Fc', 'rul']]
            val = val.drop(columns=['Fc', 'rul']).groupby(['unit', 'cycle']).transform(lambda x: x.rolling(10, 1).mean())
            val = pd.concat([val, saved_cols], axis=1)

        train[columns] -= mean; train[columns] /= std
        val[columns] -= mean; val[columns] /= std
        test[columns] -= mean; test[columns] /= std

        
        print(f"Generating parquet file {filename}")
        path = Path(args.out_path, "parquet")
        path.mkdir(exist_ok=True)
        for df, prefix in zip([train, val, test], ["train", "val", "test"]):
            if isinstance(df, pd.DataFrame):
                df.to_parquet(f"{path}/{prefix}_{filename}.parquet", engine="pyarrow")


def generate_unittest_subsample(args, vars=['X_s', 'A']) -> None:
    """ Generates parquet file in args.out_path. 
    Called by hand in dev -> Not used in project  

    Parameters
    ----------
    args : SimpleNamespace
        arguments to forward (out_path, normalization, validation...)

    Returns
    -------
    None
    """

    filename = args.files[0]
    
    filepath = os.path.join(args.out_path, filename)

    df_train, df_test = _load_data_from_file(filepath, vars=vars)
    df_train = df_train[::10000] # Huge downsample
    df_test = df_test[::10000]

    df_train = linear_piece_wise_RUL(df_train.copy(), drop_hs=False)
    df_test = linear_piece_wise_RUL(df_test.copy(), drop_hs=False)


    nosearchfor = ["unit", "cycle", "Fc", "hs", "rul"]
    columns = df_train.columns[~(df_train.columns.str.contains('|'.join(nosearchfor)))]
    # Normalization
    df_train[columns] = (df_train[columns] - df_train[columns].mean()) / df_train[columns].std()
    df_test[columns] = (df_test[columns] - df_test[columns].mean()) / df_test[columns].std()

    path = Path(args.test_path, "parquet/")
    path.mkdir(exist_ok=True)
    for df, prefix in zip([df_train, df_test], ["train", "test"]):
        if isinstance(df, pd.DataFrame):
            df.to_parquet(f"{path}/{prefix}_{filename}.parquet")



def compute_scalers(args, typ, arg="") -> Any:
    """ Compute sums for mean and std to scale multiple dataframes
        As we cannot load every file, they have to be processed one at a time
        However, the same normalization should be applied to every file

    Parameters
    ----------
    args : SimpleNamespace
        arguments to forward (out_path, normalization, validation...)
    typ : numpy type
        What precision to have for the dataframe values (np.float64, 32...)

    Returns
    -------
    columns : list of str
        which columns to scale
    means : pd.Series
        means of the columns to scale (names as index)
    stds : pd.Series
        stds of the columns to scale (names as index)

    Raises
    ------
    AssertionError
        When df is not a pd.DataFrame
    ValueError
        When neither arg or scaler is provided
        When arg is not in ['', 'minmax', 'standard']
    """

    summ = None # train, val, test
    std = None
    total_size = 0
    nosearchfor = ["unit", "cycle", "Fc", "hs", "rul"]

    print(f"Compute scaling parameters for {len(args.files)} files. Can be long")
    for i, filename in enumerate(tqdm(args.files)):
        filepath = os.path.join(args.out_path, filename)
        train, val, test = extract_validation(
            filepath=filepath,
            typ=typ,
            vars=args.subdata,
            validation=args.validation,
        )

        if args.moving_avg:
            saved_cols = train[['unit', 'cycle', 'Fc', 'rul']]
            train = train.drop(columns=['Fc', 'rul']).groupby(['unit', 'cycle']).transform(lambda x: x.rolling(10, 1).mean())
            train = pd.concat([train, saved_cols], axis=1)
            saved_cols = test[['unit', 'cycle', 'Fc', 'rul']]
            test = test.drop(columns=['Fc', 'rul']).groupby(['unit', 'cycle']).transform(lambda x: x.rolling(10, 1).mean())
            test = pd.concat([test, saved_cols], axis=1)
            saved_cols = val[['unit', 'cycle', 'Fc', 'rul']]
            val = val.drop(columns=['Fc', 'rul']).groupby(['unit', 'cycle']).transform(lambda x: x.rolling(10, 1).mean())
            val = pd.concat([val, saved_cols], axis=1)
    
        columns = val.columns[~(val.columns.str.contains('|'.join(nosearchfor)))]

        total_size += len(train) #+ len(val) + len(test)
        for i, df in enumerate([train]): #, val, test]):
            if summ is None:
                summ = df[columns].sum(axis=0)
                var = df[columns].var(axis=0) * len(df)
            else:
                summ += df[columns].sum(axis=0)
                var += df[columns].var(axis=0) * len(df)

    mean = summ / total_size
    var = var / total_size
    std = np.sqrt(var)

    return columns, mean, std


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

    unit_repartition.sort_index(inplace=True)
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


def linear_piece_wise_RUL(df: pd.DataFrame, drop_hs=True) -> pd.DataFrame:
    """ Corrects the RUL label. Uses the Health State to change RUL  
        into piece-wise linear RUL (reduces overfitting on the healthy part)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to correct.

    Returns
    -------
    df : pd.DataFrame
        Corrected dataframe

    Raises
    ------
    KeyError:
        When 'A' subset is not selected and extract_validation bypassed
    """
    healthy = df[['unit', 'hs', 'rul']] # to reduce memory footprint
    healthy = df[df.hs == 1]            # Filter on healthy
    
    mergedhealthy = healthy.merge(healthy.groupby(['unit', 'hs'])['rul'].min(), 
        how='inner', on=['unit', 'hs']) # Compute the max linear rul
    df = df.merge(mergedhealthy, how='left', on=list(df.columns[:-1]))\
        .drop(columns=['rul_x'])        # Put it back on original dataframe
    df.rul_y.fillna(df['rul'], inplace=True)    # True RUL values on NaNs
    df.drop(columns=['rul'], inplace=True)      
    df.rename({'rul_y': 'rul'}, axis=1, inplace=True)

    assert df.isna().sum().sum() == 0, "NaNs in df, on columns {}"\
        .format(df.isna().sum()[df.isna().sum() >= 1].index.values.tolist())
    
    if drop_hs:
        df.drop(columns=['hs'], inplace=True)

    return df


def extract_validation(
    filepath, typ=np.float64, vars = ['X_s', 'X_v', 'T', 'A'], validation=0.00
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

    df_train, df_test = _load_data_from_file(filepath, typ=typ, vars=vars)
    
    if 'A' in vars:
        df_train = linear_piece_wise_RUL(df_train.copy()) 
        df_test = linear_piece_wise_RUL(df_test.copy())    
    else:
        warnings.warn("'A' auxiliary variables subset was not selected."
            " RUL label will not be transformed to piece-wise linear because "
            " health state (hs) belongs to the auxiliary subset. ")


    # Percentage of datapoints across all units (engines)
    unit_repartition = df_train.groupby('unit')['rul'].count() \
        / df_train.groupby('unit')['rul'].count().sum()

    # Splits the train set into val + train
    if validation > 0:
        # Choose which engines to put in val set
        units = choose_units_for_validation(unit_repartition, validation)
        # Perform the transfer of the chosen units data
        df_val = df_train[df_train.unit.isin(units)]
        df_train = df_train[~(df_train.unit.isin(units))]

    else:
        df_val = pd.DataFrame([], columns=df_train.columns) # Empty
    
    return df_train, df_val, df_test


def _load_data_from_file(filepath, typ=np.float64, vars = ['X_s', 'X_v', 'T', 'A']):
    """Load data from source file into a dataframe.

    Parameters
    ----------
    file : str
        Source file.
    vars : list
        May contain 'X_s', 'X_v', 'T', 'A'
        W: Scenario Descriptors (always included)
        X_s: Measurements
        X_v: Virtual sensors
        T: Health Parameters
        A: Auxiliary Data
        Y: rul (always included)

    Returns
    -------
    DataFrame
        Data organized into a dataframe.
    """

    assert all([x in ['X_s', 'X_v', 'T', 'A'] for x in vars]), \
        "Wrong vars provided, choose a subset of ['X_s', 'X_v', 'T', 'A']"
    assert any([x in filepath for x in ncmapss_files]), "Incorrect file name {}"\
        .format(filepath)

    if ".h5" not in filepath:
        filepath = filepath + ".h5"

    with h5py.File(filepath, 'r') as hdf:

        dev = []
        test = []
        varnames = []

        dev.append(np.array(hdf.get('W_dev')))             
        test.append(np.array(hdf.get('W_test')))
        varnames.extend(hdf.get('W_var'))

        if 'X_s' in vars:
            dev.append(np.array(hdf.get('X_s_dev')))             
            test.append(np.array(hdf.get('X_s_test')))
            varnames.extend(hdf.get('X_s_var'))
        if 'X_v' in vars:
            dev.append(np.array(hdf.get('X_v_dev')))             
            test.append(np.array(hdf.get('X_v_test')))
            varnames.extend(hdf.get('X_v_var'))
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
        
    df_train = pd.DataFrame(data = dev, columns = varnames, dtype=typ)
    df_test = pd.DataFrame(data = test, columns = varnames, dtype=typ)

    return df_train, df_test


def generate_lmdb(args, datasets=["train", "val", "test"]) -> None:
    """ Parquet files to lmdb files
    """
    lmdb_dir = Path(f"{args.out_path}/lmdb")
    lmdb_dir.mkdir(exist_ok=True)
    lmdb_dir_files = [x for x in lmdb_dir.iterdir()]
    if len(lmdb_dir_files) > 0:
        warnings.warn(f"{lmdb_dir} is not empty. Generation will not overwrite"
            " the previously generated .lmdb files. It will append data.")
    for ds in datasets:
        filelist = list(Path(f"{args.out_path}/parquet").glob(f"{ds}*.parquet"))
        print(f"Generating {ds} lmdb with {[x.as_posix() for x in filelist]} files...")
        if filelist is not None:
            feed_lmdb(Path(f"{lmdb_dir}/{ds}.lmdb"), filelist, args)


class Line(NamedTuple): # An N-CMAPSS Line
    ds_id: int              # train, test, val
    unit_id: int            # Which unit
    win_id: int             # Window id
    settings: np.ndarray    # W
    data: np.ndarray        # X_s, X_v, T, A (not necessarily all of them)
    rul: int                # Y


def feed_lmdb(output_lmdb: Path, filelist: List[Path], args) -> None:
    patterns: Dict[str, Callable[[Line], Union[bytes, np.ndarray]]] = {
        "{}": (
            lambda line: line.data.astype(np.float32) if args.bits == 32 else line.data
        ),
        "ds_id_{}": (lambda line: "{}".format(line.ds_id).encode()),
        "unit_id_{}": (lambda line: "{}".format(line.unit_id).encode()),
        "win_id_{}": (lambda line: "{}".format(line.win_id).encode()),
        "settings_{}": (
            lambda line: line.settings.astype(np.float32)
            if args.bits == 32
            else line.settings
        ),
        "rul_{}": (lambda line: "{}".format(line.rul).encode()),
    }

    args.settings = [] # args.settings = ncmapss_datanames['W']
    args.features = []
    args.features.extend(ncmapss_datanames['W']) # We treat W as input data
    for key in ncmapss_datanames: 
        if (key == 'Y') or (key not in args.subdata): continue
        else: args.features.extend(ncmapss_datanames[key])
    

    return create_lmdb(
        filename=output_lmdb,
        iterator=process_files(filelist, args),
        patterns=patterns,
        aggs=[MinMaxAggregate(args)],
        win_length=args.win_length,
        n_features=len(args.features),
        bits=args.bits,
    )


def process_files(filelist: List[Path], args) -> Iterator[Line]:
    for filename in tqdm(filelist):
        df = pd.read_parquet(filename)
        if hasattr(args, 'skip_obs'):
            df = df[::args.skip_obs]
        args.subset = int(str(filename.stem).split('_')[-1][3])
        yield from process_dataframe(df, args)
        del df


def process_dataframe(df: pd.DataFrame, args) -> Iterator[Line]:
    win_length = (
        args.win_length[args.subset]
        if isinstance(args.win_length, dict)
        else args.win_length
    )
    for unit_id, traj in tqdm(df.groupby("unit"), leave=False, position=1):
        for i, sl in enumerate(tqdm(make_slice(traj.shape[0], win_length, 
                args.win_step), leave=False, 
                total=traj.shape[0]/args.win_step, position=2)):

            yield Line(
                ds_id=args.subset,
                unit_id=unit_id,
                win_id=i,
                settings=traj[args.settings].iloc[sl].unstack().values,
                data=traj[args.features].iloc[sl].unstack().values,
                rul=traj["rul"].iloc[sl].iloc[-1],
            )


class MinMaxAggregate:
    def __init__(self, args):
        self.args = args

    def feed(self, line: Line, i: int) -> None:
        n_features = len(self.args.features)
        min_, max_ = (
            line.data.reshape(n_features, -1).T,
            line.data.reshape(n_features, -1).T,
        )

        if i == 0:
            self.min_, self.max_ = min_.min(0), max_.max(0)
        else:
            self.min_ = np.min([self.min_, min_.min(0)], axis=0)
            self.max_ = np.max([self.max_, max_.max(0)], axis=0)

    def get(self) -> Dict[str, Union[np.ndarray, bytes]]:
        return {
            "min_sample": self.min_.astype(
                np.float32 if self.args.bits == 32 else np.float64
            ),
            "max_sample": self.max_.astype(
                np.float32 if self.args.bits == 32 else np.float64
            ),
        }
