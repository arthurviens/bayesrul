# Bayesrul

Based on https://github.com/lbasora/bnnrul

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

## Generate dataset lmdb files
Make sure to have the CMAPSS dataset at `data/CMAPSSData.zip`
Launch the scripts
```
python -m bayesrul.cmapss.generate_files
```
It will create the necessary parquet and lmdb files used later on. You can tweak the generated datasets in the `bayesrul/cmapss/generate_files.py` parameters. Parquet files are used to create lmdb files but are useless otherwise for now.

You now should have `data/cmapss/lmdb` directory.
You're all set !