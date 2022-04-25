import logging
import warnings
from typing import List, Any, NamedTuple, Callable, Dict, Iterator, Union
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import h5py
import os

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
    'A': ['unit', 'cycle', 'Fc', 'hs'],
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
    for filename in args.files:
        logging.info("**** %s ****" % filename)
        logging.info("normalization = " + args.normalization)
        logging.info("validation = " + str(args.validation))

        filepath = os.path.join(args.out_path, filename)

        print("Extracting dataframes...")
        df_train, df_val, df_test = extract_validation(
            filepath=filepath,
            vars=args.subdata,
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
    filepath, vars = ['X_s', 'X_v', 'T', 'A'], validation=0.00
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


def _load_data_from_file(filepath, vars = ['X_s', 'X_v', 'T', 'A']):
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
        
    df_train = pd.DataFrame(data = dev, columns = varnames)
    df_test = pd.DataFrame(data = test, columns = varnames)
    
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
        print(f"Generating {ds} lmdb ...")
        filelist = list(Path(f"{args.out_path}/parquet").glob(f"{ds}*.parquet"))
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

    args.settings = ncmapss_datanames['W']
    args.features = []
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
