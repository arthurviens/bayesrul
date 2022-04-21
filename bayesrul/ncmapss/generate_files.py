from types import SimpleNamespace
from .preprocessing import generate_parquet


if __name__ == "__main__":
    args = SimpleNamespace(
        out_path = "data/ncmapss/",
        normalization = "min-max",
        validation = 0.2,
        files = ["N-CMAPSS_DS02-006"],
        bits = 32,
    )

    generate_parquet(args)