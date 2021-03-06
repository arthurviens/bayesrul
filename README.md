# Bayesrul


## Setup 

Clone the repository
```
git clone git@github.com:arthurviens/bayesrul.git
cd bayesrul
```

Create a new environment (python 3.8 and 3.9 supported)
```
conda create -n bayesrul python=3.8
conda activate bayesrul
```

Install poetry to manage all the dependencies for you
```
pip install poetry
```

Use poetry to install dependencies
```
poetry install
```

TyXe has to be installed by hand, as it is not on PyPI
```
pip install git+https://github.com/TyXe-BDL/TyXe.git
```

## Generate dataset lmdb files for N-CMAPSS
Make sure to have the CMAPSS dataset files at `data/ncmapss/N-CMAPSS_DS02-006.h5` (you can use other subsets by modifying arguments in `bayesful/ncmapss/generate_files.py`)

Launch the script
```
python -m bayesrul.ncmapss.generate_files
```
It will create the necessary parquet and lmdb files used later on. Parquet files are used to create lmdb files but are useless otherwise for now.

You now should have `data/ncmapss/lmdb` directory, with all that is needed inside.

You can now launch a training 
```
python -m bayesrul.ncmapss.train_model --bayesian --archi inception --model-name --guide normal testname
```