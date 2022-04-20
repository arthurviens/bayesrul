from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from types import SimpleNamespace
from .preprocessing import generate_parquet, generate_lmdb


if __name__ == "__main__":
    args = SimpleNamespace(
        out_path="data/cmapss/",
        normalization="min-max",
        validation=0.2,
        subsets=["FD001"], #,"FD002", "FD003", "FD004"],
        # win_length={"FD001": 30, "FD002": 20, "FD003": 30, "FD004": 15}, #variable length per subset only ok for LSTM models
        # win_length={
        #     1: 18,
        #     2: 18,
        #     3: 18,
        #     4: 18,
        # },  # variable length per subset fixed for Linear/Conv models
        win_length=25,
        win_step=1,
        settings=["setting_1", "setting_2", "setting_3"],
        features=[
            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_7",
            "sensor_8",
            "sensor_9",
            "sensor_11",
            "sensor_12",
            "sensor_13",
            "sensor_14",
            "sensor_15",
            "sensor_17",
            "sensor_20",
            "sensor_21",
        ],
        bits=32,
    )

    generate_parquet(args)
    generate_lmdb(args)