import pytest

import pandas as pd
import numpy as np

from bayesrul.deprecated.cmapss.preprocessing import normalize_cmapss


def test_normalize_cmapss():
    df = pd.DataFrame(
        [[1, 1, -2, 0], [2, -1, -1, 1], [3, 0, 1, 0.5]], 
        columns=["id", "sensor1", "sensor2", "setting1"]
    )

    df_standard_calc = df.copy()
    _ = normalize_cmapss(df_standard_calc, arg='standard')

    df_standard_true = pd.DataFrame(
        [[1, 1.224745, -1.069045, -1.224745], [2, -1.224745, -0.267261,
            1.224745], [3, 0, 1.336306, 0]], 
        columns=["id", "sensor1", "sensor2", "setting1"]
    )

    df_minmax_calc = df.copy()
    scaler = normalize_cmapss(df_minmax_calc, arg='min-max')

    df_minmax_true = pd.DataFrame(
        [[1, 1.0, 0, 0], [2, 0, 0.333333,
            1.0], [3, 0.5, 1.0, 0.5]], 
        columns=["id", "sensor1", "sensor2", "setting1"]
    )

    _ = normalize_cmapss(df, scaler=scaler)

    assert np.linalg.norm(df_standard_calc - df_standard_true) <= 1e-5
    assert np.linalg.norm(df_minmax_calc - df_minmax_true) <= 1e-5
    assert np.linalg.norm(df - df_minmax_true) <= 1e-5

    with pytest.raises(ValueError):
        normalize_cmapss(df, arg="itshouldfail")
    with pytest.raises(ValueError):
        normalize_cmapss(df)

