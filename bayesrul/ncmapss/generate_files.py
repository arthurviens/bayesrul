from types import SimpleNamespace
from .preprocessing import generate_parquet, generate_lmdb


if __name__ == "__main__":
    args = SimpleNamespace(
        out_path = "data/ncmapss/",
        normalization = "min-max",
        validation = 0.2,
        files = ["N-CMAPSS_DS02-006"],
        subdata = ['X_s', 'X_v', 'T', 'A'],
        win_length=25,
        win_step=10,
        skip_obs=10,
        bits = 32,
    )

    #generate_parquet(args)
    generate_lmdb(args)