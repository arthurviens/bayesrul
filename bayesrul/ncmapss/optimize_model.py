from bayesrul.inference import (
    vi_bnn,
    dnn,
    mc_dropout,
    deep_ensemble
)

from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.utils.miscellaneous import simple_cull, select_pareto
from pathlib import Path

import torch
import pyro
import argparse
import optuna
import json

debug = False
EPOCHS = 150 if not debug else 2
GPU = 2
device = torch.device(f'cuda:{GPU}')


def mfvi_objective(trial: optuna.trial.Trial) -> float:
    pyro.clear_param_store()

    prior_scale = trial.suggest_float("prior_scale", 1e-4, 1, log=True)
    q_scale = trial.suggest_float("q_scale", 1e-4, 1e-1, log=True)
    fit_context = trial.suggest_categorical("fit_context", ['lrt', 'flipout']) 
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    guide = "normal"
    optimizer = trial.suggest_categorical("optimizer", ['adam', 'nadam', 'sgd'])
    args.archi = trial.suggest_categorical("args.archi", 
            ['linear', 'conv', 'inception', 'bigception'])
    args.activation = trial.suggest_categorical("args.activation", 
        ['leaky_relu', 'tanh', 'relu'])
    args.pretrain = 5 #trial.suggest_categorical("args.pretrain", [0, 10])
    args.model_name = f'mfvi{trial.number:03d}'
        
    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    
    hyperparams = {
        'bias' : True,
        'prior_loc' : 0,
        'prior_scale' : prior_scale,
        'q_scale' : q_scale,
        'fit_context' : fit_context,
        'lr' : lr,
        'guide' : guide,
        'optimizer' : optimizer,
        'pretrain_file' : None,
        'pretrain': args.pretrain,
        'trial_id' : trial.number,
        'device' : device,
    }

    monitors = ["mse/val", "rmsce/val"]
    inference = vi_bnn.VI_BNN(args, data, hyperparams, GPU=GPU, studying=True)
    res = inference.fit(EPOCHS, monitors=monitors)


    return res[0], res[1]

    
def radial_objective(trial: optuna.trial.Trial) -> float:
    pyro.clear_param_store()

    prior_scale = trial.suggest_float("prior_scale", 1e-4, 1, log=True)
    q_scale = trial.suggest_float("q_scale", 1e-4, 1e-1, log=True)
    fit_context = trial.suggest_categorical("fit_context", ['lrt', 'flipout']) 
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    guide = "radial"
    optimizer = trial.suggest_categorical("optimizer", ['adam', 'nadam', 'sgd'])
    args.archi = trial.suggest_categorical("args.archi", 
            ['linear', 'conv', 'inception', 'bigception'])
    args.activation = trial.suggest_categorical("args.activation", 
        ['leaky_relu', 'tanh', 'relu'])
    args.pretrain = 5 # trial.suggest_categorical("args.pretrain", [0, 10])
    args.model_name = f'radial{trial.number:03d}'
        
    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    
    hyperparams = {
        'bias' : True,
        'prior_loc' : 0,
        'prior_scale' : prior_scale,
        'q_scale' : q_scale,
        'fit_context' : fit_context,
        'lr' : lr,
        'guide' : guide,
        'optimizer' : optimizer,
        'pretrain': args.pretrain,
        'pretrain_file' : None,
        'trial_id' : trial.number,
        'device' : device,
    }

    monitors = ["mse/val", "rmsce/val"]
    inference = vi_bnn.VI_BNN(args, data, hyperparams, GPU=GPU, studying=True)
    res = inference.fit(EPOCHS, monitors=monitors)

    return res[0], res[1]

    
def lowrank_objective(trial: optuna.trial.Trial) -> float:
    pyro.clear_param_store()

    prior_scale = trial.suggest_float("prior_scale", 1e-4, 1, log=True)
    q_scale = trial.suggest_float("q_scale", 1e-4, 1e-1, log=True)
    fit_context = trial.suggest_categorical("fit_context", ['lrt', 'flipout']) 
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    guide = "lowrank"
    optimizer = trial.suggest_categorical("optimizer", ['adam', 'nadam', 'sgd'])
    args.archi = trial.suggest_categorical("args.archi", 
            ['inception', 'bigception'])
    args.activation = trial.suggest_categorical("args.activation", 
        ['leaky_relu', 'tanh', 'relu'])
    args.pretrain = 5 #trial.suggest_categorical("args.pretrain", [0, 10])
    args.model_name = f'lowrank{trial.number:03d}'
        
    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    
    hyperparams = {
        'bias' : True,
        'prior_loc' : 0,
        'prior_scale' : prior_scale,
        'q_scale' : q_scale,
        'fit_context' : fit_context,
        'lr' : lr,
        'guide' : guide,
        'optimizer' : optimizer,
        'pretrain' : args.pretrain,
        'pretrain_file' : None,
        'trial_id' : trial.number,
        'device' : device,
    }

    monitors = ["mse/val", "rmsce/val"]
    inference = vi_bnn.VI_BNN(args, data, hyperparams, GPU=GPU, studying=True)
    res = inference.fit(EPOCHS, monitors=monitors)

    return res[0], res[1]
    
    
def mcdropout_objective(trial: optuna.trial.Trial) -> float:
    raise NotImplementedError('Not implemented yet')
    
def deepensemble_objective(trial: optuna.trial.Trial) -> float:
    raise NotImplementedError('Not implemented yet')
    
def heteroscedasticnn_objective(trial: optuna.trial.Trial) -> float:
    raise NotImplementedError('Not implemented yet')
    
def mcmc_objective(trial: optuna.trial.Trial) -> float:
    raise NotImplementedError('Not implemented yet')
    

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
                    default='mfvi',
                    metavar='MODEL',
                    help='Model to study. (default: mfvi)',
                    required=True)
    parser.add_argument('--study-name',
                    type=str,
                    default='study',
                    metavar='STUDY_NAME',
                    help='Name of this specific study. (default: study)',
                    required=True)

    args = parser.parse_args()
    
    if args.model == "mfvi":
        objective = mfvi_objective
    elif args.model == "radial":
        objective = radial_objective
    elif args.model == "lowrank":
        objective = lowrank_objective
    elif args.model == "mcmc":
        objective = mcmc_objective
    elif args.model == "mc_dropout":
        objective = mcdropout_objective
    elif args.model == "deep_ensemble":
        objective = deepensemble_objective
    elif args.model == "hetero_nn":
        objective = heteroscedasticnn_objective
    else:
        raise ValueError(
            f"Unknown model '{args.model}'. Try 'mfvi', 'radial', 'lowrank"
            "'mcmc', 'mc_dropout', 'deep_ensemble', 'hetero_nn'."
            )

    sampler = optuna.samplers.RandomSampler()
    #sampler = optuna.samplers.NSGAIIMultiObjectiveSampler()

    path = Path("results/ncmapss/studies")
    path.mkdir(exist_ok=True)

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name=args.study_name, 
        sampler=sampler,
        storage="sqlite:///"+path.as_posix()+"/optimization.db",
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=300, 
        timeout=None,
        catch=(RuntimeError,),
    )
    
    print("Number of finished trials: {}".format(len(study.trials)))

    df = study.trials_dataframe()
    df.dropna(axis=0, inplace=True)
    pareto, dominated = simple_cull(df[['values_0', 'values_1']].values.tolist())
    pareto = select_pareto(df, pareto)

    p = Path('results/ncmapss/best_models/', args.study_name)
    p.mkdir(exist_ok=True)
    pareto.sort_values('values_0', inplace=True)
    pareto.reset_index(drop=True, inplace=True)
    for i, row in pareto.iterrows():
        n = row['number']
        #string = json.dumps(, indent=4)
        with open(Path(p, f'{i:03d}.json').as_posix(), 'w') as f:
            params = study.trials[n].params
            params['value_0'] = study.trials[n].values[0]
            params['value_1'] = study.trials[n].values[1]
            json.dump(params, f, indent=4)
