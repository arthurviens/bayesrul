from types import SimpleNamespace
from .preprocessing import generate_parquet, generate_lmdb, generate_unittest_subsample


if __name__ == "__main__":
    args = SimpleNamespace(
        out_path = "data/ncmapss/",
        test_path = "tests/",
        normalization = "min-max",
        validation = 0.2,
        files = ["N-CMAPSS_DS02-006", "N-CMAPSS_DS03-012"],
        subdata = ['X_s', 'A'],
        win_length=25,  # Window size
        win_step=10,    # Window step
        skip_obs=10,    # How much to downsample the huge dataset
        bits = 32,      # Size of numbers in memory
    )

    generate_parquet(args)
    generate_lmdb(args)
    
    #args.files = ["N-CMAPSS_DS02-006"]
    #generate_unittest_subsample(args) # To create unit test parquets