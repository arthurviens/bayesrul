from types import SimpleNamespace
from .preprocessing import generate_parquet, generate_lmdb, generate_unittest_subsample


if __name__ == "__main__":
    args = SimpleNamespace(
        out_path = "data/ncmapss/",
        test_path = "tests/",
        normalization = "standard",
        validation = 0.10,
        files = [
            #"N-CMAPSS_DS01-005",
            "N-CMAPSS_DS02-006", 
            "N-CMAPSS_DS03-012", 
            "N-CMAPSS_DS04", 
            "N-CMAPSS_DS05"
        ],
        subdata = ['X_s', 'A'],
        moving_avg=True, # Smooth the values of the sensors
        win_length=30,  # Window size
        win_step=10,    # Window step
        skip_obs=10,    # How much to downsample the huge dataset
        bits=32,      # Size of numbers in memory
    )

    #generate_parquet(args)
    generate_lmdb(args)
    
    #args.files = ["N-CMAPSS_DS02-006"]
    #generate_unittest_subsample(args) # To create unit test parquets