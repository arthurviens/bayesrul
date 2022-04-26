from types import SimpleNamespace
from .preprocessing import generate_parquet, generate_lmdb


if __name__ == "__main__":
    args = SimpleNamespace(
        out_path = "data/ncmapss/",
        normalization = "min-max",
        validation = 0.2,
        files = ["N-CMAPSS_DS02-006"],
        subdata = ['X_s', 'A'],
        win_length=25,  # Window size
        win_step=10,    # Window step
        skip_obs=10,    # How much to downsample the huge dataset
        bits = 32,      # Size of numbers in memory
    )

    generate_parquet(args)
    generate_lmdb(args)