import pytest

import pandas as pd
import numpy as np

from bayesrul.ncmapss.preprocessing import normalize_ncmapss
from bayesrul.ncmapss.preprocessing import choose_units_for_validation
from bayesrul.ncmapss.preprocessing import linear_piece_wise_RUL


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