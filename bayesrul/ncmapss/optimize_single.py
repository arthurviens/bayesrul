from bayesrul.inference import (
    vi_bnn,
    dnn,
    mc_dropout,
    deep_ensemble
)

from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from pathlib import Path

import torch
import pyro
import argparse
import optuna

debug = False
EPOCHS = 150 if not debug else 2


"""
Launches Optuna search for single objective optimization (even if the training 
loss is not MSE, the validation loss on which we optimize across trials is MSE)

All implemented models can be launched for optimization
Trials are stored into a study 

Example on how to launch:
python -m bayesrul.ncmapss.optimize_single --model lrt --study-name LRT --GPU 0 --sampler TPE
"""


"""
'Objective' functions are implemented, launched by Optuna for hyperparam search
They can:
    - Define which parameters and which space to search
    - Initialize data loader 
    - Initializes model
    - Fits the model
    - Get the result
"""

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

    monitors = ["mse/val"]
    inference = vi_bnn.VI_BNN(args, data, hyperparams, GPU=GPU, studying=True)
    res = inference.fit(EPOCHS, monitors=monitors)


    return res[0]

def lrt_objective(trial: optuna.trial.Trial) -> float:
    pyro.clear_param_store()

    prior_scale = trial.suggest_float("prior_scale", 1e-4, 1, log=True)
    q_scale = trial.suggest_float("q_scale", 1e-4, 1e-1, log=True)
    fit_context = 'lrt'
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    guide = "normal"
    optimizer = 'adam'
    args.archi = trial.suggest_categorical("args.archi", 
            ['inception', 'bigception'])
    args.activation = 'relu'
    args.pretrain = 5 
    args.model_name = f'lrt_{trial.number:03d}'
        
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

    monitors = ["mse/val"]
    inference = vi_bnn.VI_BNN(args, data, hyperparams, GPU=GPU, studying=True)
    res = inference.fit(EPOCHS, monitors=monitors)

    return res[0]
    

def flipout_objective(trial: optuna.trial.Trial) -> float:
    pyro.clear_param_store()

    prior_scale = trial.suggest_float("prior_scale", 1e-4, 1, log=True)
    q_scale = trial.suggest_float("q_scale", 1e-4, 1e-1, log=True)
    fit_context = 'flipout'
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    guide = "normal"
    optimizer = 'adam'
    args.archi = trial.suggest_categorical("args.archi", 
            ['inception', 'bigception'])
    args.activation = 'relu'
    args.pretrain = 5 
    args.model_name = f'flipout_{trial.number:03d}'
        
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

    monitors = ["mse/val"]
    inference = vi_bnn.VI_BNN(args, data, hyperparams, GPU=GPU, studying=True)
    res = inference.fit(EPOCHS, monitors=monitors)

    return res[0]
    

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

    monitors = ["mse/val"]
    inference = vi_bnn.VI_BNN(args, data, hyperparams, GPU=GPU, studying=True)
    res = inference.fit(EPOCHS, monitors=monitors)

    return res[0]

    
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

    monitors = ["mse/val"]
    inference = vi_bnn.VI_BNN(args, data, hyperparams, GPU=GPU, studying=True)
    res = inference.fit(EPOCHS, monitors=monitors)

    return res[0]
    
    
def mcdropout_objective(trial: optuna.trial.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    p_dropout = trial.suggest_float("p_dropout", 0.20, 0.85)
    args.archi = trial.suggest_categorical("args.archi", 
            ['inception', 'bigception'])
    args.activation = trial.suggest_categorical("args.activation", 
        ['leaky_relu', 'tanh', 'relu'])
    args.model_name = f'mcdropout_{trial.number:03d}'

    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    
    hyperparams = {
        'bias' : True,
        'lr' : lr,
        'p_dropout' : p_dropout,
        'trial_id' : trial.number,
        'device' : device,
    }

    monitors = ["mse/val"]
    inference = mc_dropout.MCDropout(args, data, p_dropout,
                    hyperparams=hyperparams, GPU=GPU, studying=True)
    res = inference.fit(EPOCHS, monitors=monitors)

    return res
    
def deepensemble_objective(trial: optuna.trial.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    n_models = trial.suggest_categorical("n_models", [3, 5, 7, 10])
    args.archi = trial.suggest_categorical("args.archi", 
            ['inception', 'bigception'])
    args.activation = trial.suggest_categorical("args.activation", 
        ['leaky_relu', 'tanh', 'relu'])
    args.model_name = f'deepensemble_{trial.number:03d}'

    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    
    hyperparams = {
        'bias' : True,
        'lr' : lr,
        'trial_id' : trial.number,
        'device' : device,
    }

    monitors = ["mse/val", "rmsce/val"]
    inference = deep_ensemble.DeepEnsemble(args, data, n_models,
                    hyperparams=hyperparams, GPU=GPU, studying=True)
    res = inference.fit(EPOCHS, monitors=monitors)

    return res
    
def heteroscedasticnn_objective(trial: optuna.trial.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    args.archi = trial.suggest_categorical("args.archi", 
            ['inception', 'bigception', 'linear', 'conv'])
    args.activation = trial.suggest_categorical("args.activation", 
        ['leaky_relu', 'tanh', 'relu'])
    args.model_name = f'heteronn_{trial.number:03d}'

    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    
    hyperparams = {
        'bias' : True,
        'lr' : lr,
        'trial_id' : trial.number,
        'device' : device,
    }

    monitors = ["mse/val"]
    inference = dnn.HeteroscedasticDNN(args, data,
                    hyperparams=hyperparams, GPU=GPU, studying=True)
    res = inference.fit(EPOCHS, monitors=monitors)

    return res
    
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
    parser.add_argument('--GPU',
                    type=int,
                    default=1,
                    metavar='GPU',
                    help='Which GPU to use (0,1,2). (default: 1)',
                    required=True)
    parser.add_argument('--sampler',
                    type=str,
                    default='random',
                    metavar='SAMPLER',
                    help='Whether to use random sampling (0,1,2).',
                    required=True)

    args = parser.parse_args()


    GPU = args.GPU
    device = torch.device(f'cuda:{GPU}')
    
    if args.model == "mfvi": # LRT or FLIPOUT
        objective = mfvi_objective
    elif args.model == "lrt":
        objective = lrt_objective
    elif args.model == "flipout":
        objective = flipout_objective
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
            f"Unknown model '{args.model}'. Try 'mfvi', 'lrt' 'flipout', "
            "'radial', 'lowrank', 'mcmc', 'mc_dropout', 'deep_ensemble', "
            "'hetero_nn'."
            )

    if args.sampler.upper() == 'RANDOM':
        sampler = optuna.samplers.RandomSampler()
    elif args.sampler.upper() == 'TPE':
        sampler = optuna.samplers.TPESampler()
    else:
        raise ValueError(f"Unknow sampler {args.sampler}. Choose RANDOM or NSGAII")

    pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=10)

    path = Path("results/ncmapss/studies/single_obj")
    path.mkdir(exist_ok=True, parents=True)

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name, 
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///"+path.as_posix()+"/studies.db",
        load_if_exists=True,
    )

    print(f"Sampler : {study.sampler}")
    
    study.optimize(
        objective,
        n_trials=150, 
        timeout=None,
        catch=(RuntimeError,),
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    df = study.trials_dataframe()
    df.dropna(axis=0, inplace=True)

    print(df)