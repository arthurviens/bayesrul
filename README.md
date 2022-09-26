# Bayesrul


## Setup 

Clone the repository
```
git clone git@github.com:arthurviens/bayesrul.git
cd bayesrul
```

Use poetry to install dependencies
```
poetry install
```

## Generate dataset lmdb files for N-CMAPSS
Make sure to have the CMAPSS dataset files at `data/ncmapss/N-CMAPSS_DS02-006.h5` (you can use other subsets by modifying arguments in `bayesful/ncmapss/generate_files.py`)

Launch the script
```
poetry run python -m bayesrul.ncmapss.generate_files
```
It will create the necessary parquet and lmdb files used later on. Parquet files are used to create lmdb files but are useless otherwise for now.

You now should have `data/ncmapss/lmdb` directory, with all that is needed inside.


## Train a model

You can now launch a training (for example a BNN) 
```
poetry run python -m bayesrul.ncmapss.train_model --bayesian --archi inception --guide normal --GPU 0 --model-name My_Model 
```

Or a frequentist model by removing `--bayesian`
```
poetry run python -m bayesrul.ncmapss.train_model --archi inception --GPU 0 --model-name My_Model 
```


Or only launch the model on test set if it has already been trained

```
poetry run python -m bayesrul.ncmapss.train_model --bayesian --GPU 0 --model-name My_Model --test 
```

## Launch Optuna searches

It's possible to launch a hyperparameter search for LRT on GPU 0
```
poetry run python -m bayesrul.ncmapss.optimize_single --model lrt --study-name LRT --sampler TPE --GPU 0 
```

In a JSON you can save the best parameter the search tried in the directory and file:
` results/ncmapss/best_models/LRT/000.json `

With such adictionary of parameters, it is possible to launch a training with these best found parameters. It will read the file and initialize a model accordingly before training.
```
poetry run python -m bayesrul.ncmapss.train_best_models --model LRT --GPU 0
```
