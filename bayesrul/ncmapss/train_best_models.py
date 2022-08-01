from pathlib import Path 
import pandas as pd 

import argparse
import json

from bayesrul.inference.vi_bnn import VI_BNN
from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.inference.dnn import HomoscedasticDNN, HeteroscedasticDNN
from bayesrul.inference.mc_dropout import MCDropout
from bayesrul.inference.deep_ensemble import DeepEnsemble

DEBUG = False
EPOCHS = 2 if DEBUG else 500


def bayesian_or_not(s):
    if s.upper() in ["MFVI", "RADIAL", "LOWRANK", "LRT", "FLIPOUT"]:
        return True
    elif s.upper() in ["MC_DROPOUT", "DEEP_ENSEMBLE", "HETERO_NN"]:
        return False
    else:
        raise ValueError(f"Unknow model {s}. Choose from mfvi, lrt, flipout, "
            "radial, lowrank, mc_dropout, deep_ensemble, hetero_nn ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayesrul benchmarking')
    parser.add_argument('--data-path',
                    type=str,
                    default='data/ncmapss',
                    metavar='DATA',
                    help='Directory where to find the data')
    parser.add_argument('--out-path',
                    type=str,
                    default='results/ncmapss/',
                    metavar='OUT',
                    help='Directory where to store models and logs')
    args = parser.parse_args()

    models = ['LRT']
    path = "results/ncmapss/best_models"

    for model in models:
        path = Path(path, model)
        ls = sorted(list(path.glob('*.json')))

        with open(ls[0], 'r') as f:
            hyp = json.load(f)
            hyp['GPU'] = 0
        try:
            del hyp['value_0']; del hyp['value_1']
        except KeyError:
            pass

        args.model_name = model
        data = NCMAPSSDataModule(args.data_path, batch_size=10000)

        if bayesian_or_not(model):
            module = VI_BNN(args, data, hyp)
            module.fit(EPOCHS)
        else:
            if model == "MC_DROPOUT":
                p_dropout = hyp['dropout']
                module = MCDropout(args, data, p_dropout, hyp)
            elif model == "DEEP_ENSEMBLE":
                module = DeepEnsemble(args, data, 5, hyp)
            elif model == "HETERO_NN":
                module = HeteroscedasticDNN(args, data, hyp)
            else:
                raise ValueError(f"Wrong model {model}. Available : MFVI, "
                    "RADIAL, LOWRANK, MC_DROPOUT, DEEP_ENSEMBLE, HETERO_NN")
            ... # TODO
        print(hyp)