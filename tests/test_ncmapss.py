import pytest

import pandas as pd
import numpy as np

from bayesrul.ncmapss.preprocessing import normalize_ncmapss
from bayesrul.ncmapss.preprocessing import choose_units_for_validation
from bayesrul.ncmapss.preprocessing import linear_piece_wise_RUL
from bayesrul.ncmapss.preprocessing import process_files
from bayesrul.ncmapss.preprocessing import process_dataframe

from types import SimpleNamespace
from pathlib import Path

args = SimpleNamespace(
    out_path = "data/ncmapss/",
    test_path = "tests/",
    normalization = "min-max",
    validation = 0.2,
    files = ["N-CMAPSS_DS02-006"],
    subdata = ['X_s', 'A'],
    win_length=25,  # Window size
    win_step=10,    # Window step
    bits = 32,      # Size of numbers in memory
)


def test_normalize_ncmapss():
    df = pd.DataFrame(
        [[1, 1, -2, 0], [2, -1, -1, 1], [3, 0, 1, 0.5]], 
        columns=["rul", "sensor1", "sensor2", "setting1"]
    )

    df_standard_calc = df.copy()
    _ = normalize_ncmapss(df_standard_calc, arg='standard')

    df_standard_true = pd.DataFrame(
        [[1, 1.224745, -1.069045, -1.224745], [2, -1.224745, -0.267261,
            1.224745], [3, 0, 1.336306, 0]], 
        columns=["rul", "sensor1", "sensor2", "setting1"]
    )

    df_minmax_calc = df.copy()
    scaler = normalize_ncmapss(df_minmax_calc, arg='min-max')

    df_minmax_true = pd.DataFrame(
        [[1, 1.0, 0, 0], [2, 0, 0.333333,
            1.0], [3, 0.5, 1.0, 0.5]], 
        columns=["rul", "sensor1", "sensor2", "setting1"]
    )

    _ = normalize_ncmapss(df, scaler=scaler)

    assert np.linalg.norm(df_standard_calc - df_standard_true) <= 1e-5
    assert np.linalg.norm(df_minmax_calc - df_minmax_true) <= 1e-5
    assert np.linalg.norm(df - df_minmax_true) <= 1e-5

    with pytest.raises(ValueError):
        normalize_ncmapss(df, arg="itshouldfail")
    with pytest.raises(ValueError):
        normalize_ncmapss(df)


def test_choose_units_for_validation():
    # Test error throws
    rep = pd.Series([0.05, 0.05, 0.1, 0.2, 0.3])
    with pytest.raises(AssertionError):
        choose_units_for_validation(rep, 0.5)

    rep = pd.Series([0.5, 0.5, 0])
    with pytest.raises(AssertionError):
        choose_units_for_validation(rep, 0.5)

    # Test function returns
    rep = pd.Series([0.05, 0.05, 0.1, 0.2, 0.3, 0.3])
    units = choose_units_for_validation(rep, 0.2)
    assert units[0] == 3

    rep = pd.Series([0.09, 0.11, 0.4, 0.4])
    units = choose_units_for_validation(rep, 0.2)
    assert len(units) == 2
    assert (units[0]==0) & (units[1]==1) 

    rep = pd.Series([0.31, 0.28, 0.31, 0.1])
    units = choose_units_for_validation(rep, 0.2)
    assert units[0] == 1


def test_linear_piece_wise_RUL():
    df = pd.DataFrame()
    with pytest.raises(KeyError):
        linear_piece_wise_RUL(df)

    columns = ['data', 'unit', 'hs', 'rul']
    df = pd.DataFrame(
        data = np.array([np.random.normal(size=10),
                        [11]*10, 
                        [1]*5 + [0] * 5, 
                        list(range(10, 0, -1))]).T,
        columns=columns
    )
    
    df = pd.concat([df, 
        pd.DataFrame(
            data = np.array([np.random.normal(size=10),
                    [15]*10, 
                    [1]*3 + [0] * 7, 
                    list(range(10, 0, -1))]).T,
            columns=columns
        )
    ]).reset_index(drop=True)

    new_df = linear_piece_wise_RUL(df)

    assert all(new_df['rul'] == [6.0]*5 + list(range(5, 0, -1))
                            + [8.0]*3 + list(range(7, 0, -1)))

    assert 'hs' not in new_df.columns


def test_data_access():
    pq_path = Path(args.test_path, 'parquet')
    df_test = pd.read_parquet(Path(pq_path, 'test_N-CMAPSS_DS02-006.parquet'))
    df_train = pd.read_parquet(Path(pq_path, 'train_N-CMAPSS_DS02-006.parquet'))
    
    assert df_test.shape == (126, 23)  # Change this
    assert df_train.shape == (527, 23) # Change this


def test_process_dataframe():
    pq_path = Path(args.test_path, 'parquet')
    args.subset=2
    args.settings=[]
    args.features = ['alt', 'Mach', 'TRA', 'T2', 'T24', 'T30', 'T48', 'T50', 
            'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 
            'Wf', 'unit', 'cycle', 'Fc', 'hs']
    
    df = pd.read_parquet(Path(pq_path, 'train_N-CMAPSS_DS02-006.parquet'))

    for line in process_dataframe(df, args):
        assert line.ds_id == 2
        assert line.unit_id == 2.0
        assert line.win_id == 0
        assert len(line.settings) == 0
        assert np.linalg.norm(line.data) - 67.347903 <= 1e-7
        assert line.rul == 53.0
        
        break



def test_process_files():
    pq_path = Path(args.test_path, 'parquet')
    
    args.subset=2
    args.settings=[]
    args.features = ['alt', 'Mach', 'TRA', 'T2', 'T24', 'T30', 'T48', 'T50', 
            'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 
            'Wf', 'unit', 'cycle', 'Fc', 'hs']
    
    
    results = pd.DataFrame(columns = ['ds', 'unit', 'win', 'norm', 'rul'])
    for line in process_files(pq_path.iterdir(), args):
        s = pd.DataFrame([[line.ds_id, line.unit_id, line.win_id, 
            np.linalg.norm(line.data), line.rul]], 
            columns=['ds', 'unit', 'win', 'norm', 'rul'])
        results = pd.concat([results, s], axis=0, ignore_index=True)

    assert np.linalg.norm(results.loc[0].values
        - np.array([2, 11.0, 0, 85.13037257472445, 37.0])) <= 1e-7
    assert np.linalg.norm(results.loc[results.shape[0] - 1].values
        - np.array([2, 20.0, 6, 297.4568, 0])) <= 1e-7