from pathlib import Path 
import pandas as pd 

import argparse
import json
import torch

from bayesrul.inference.vi_bnn import VI_BNN
from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.inference.dnn import HomoscedasticDNN, HeteroscedasticDNN
from bayesrul.inference.mc_dropout import MCDropout
from bayesrul.inference.deep_ensemble import DeepEnsemble

DEBUG = False
EPOCHS = 2 if DEBUG else 500

"""
For a given model ("FLIPOUT" for example), retrieves the best parameters in the file
results/ncmapss/best_models/FLIPOUT/000.json and trains a model according
to the hyperparameters stored in this file

Deep ensembles train more epochs, because training examples are asplit among ensembles
"""


def bayesian_or_not(s):
    #s = '_'.join(s.split('_')[:-1])
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
    parser.add_argument('--model',
                    type=str,
                    default='LRT',
                    metavar='MODEL',
                    required=True,
                    help='Model and name (ex: LRT_001)')
    parser.add_argument('--GPU',
                    type=int,
                    default='results/ncmapss/',
                    metavar='GPU',
                    required=True,
                    help='GPU index (ex: 1)')
    
    args = parser.parse_args()


    path = "results/ncmapss/best_models"

    model = '_'.join(args.model.split('_')[:-1])
    model_path = Path(path, model)
    ls = sorted(list(model_path.glob('*.json')))

    with open(ls[0], 'r') as f:
        hyp = json.load(f)
    try:
        del hyp['value_0']; del hyp['value_1']
    except KeyError:
        pass

    for i in range(1):
        #args.model_name = model + f"_{i:03d}"
        args.model_name = args.model
        data = NCMAPSSDataModule(args.data_path, batch_size=10000)

        if bayesian_or_not(model):
            hyp['pretrain'] = 5
            module = VI_BNN(args, data, hyp, GPU=args.GPU)
        else:
            if model == "MC_DROPOUT":
                p_dropout = hyp['p_dropout']
                module = MCDropout(args, data, hyp['p_dropout'], hyp, GPU=args.GPU)
            elif model == "DEEP_ENSEMBLE":
                EPOCHS = EPOCHS * 3
                module = DeepEnsemble(args, data, hyp['n_models'], hyp, GPU=args.GPU)
            elif model == "HETERO_NN":
                module = HeteroscedasticDNN(args, data, hyp, GPU=args.GPU)
            else:
                raise ValueError(f"Wrong model {model}. Available : MFVI, "
                    "RADIAL, LOWRANK, MC-DROPOUT, DEEP-ENSEMBLE, HETERO-NN")
        
        
        module.fit(EPOCHS)
        module.test()
        try:
            module.epistemic_aleatoric_uncertainty(device=torch.device('cpu'))
        except Exception:
            pass
            