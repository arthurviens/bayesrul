import pytorch_lightning as pl

from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.lightning_wrappers.bayesian import VIBnnWrapper
from bayesrul.lightning_wrappers.frequentist import DnnPretrainWrapper
from optuna.integration import PyTorchLightningPruningCallback
from pathlib import Path

import torch
import pyro
import argparse
import optuna

debug = False
EPOCHS = 200 if not debug else 2
device = torch.device('cuda:2')

def objective(trial: optuna.trial.Trial) -> float:
    pyro.clear_param_store()

    #bias = trial.suggest_categorical("bias", [True, False])
    prior_loc = 0 #trial.suggest_float("prior_loc", -0.2, 0.2)
    prior_scale = trial.suggest_float("prior_scale", 1e-4, 1, log=True)
    likelihood_scale = 0 #trial.suggest_float("likelihood_scale", 1e-2, 1e2, log=True)
    q_scale = trial.suggest_float("q_scale", 1e-4, 1e-1, log=True)
    fit_context = trial.suggest_categorical("fit_context", ['lrt', 'flipout']) 
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    num_particles = 1# trial.suggest_categorical("num_particles", [1, 3])
    guide_base = trial.suggest_categorical("guide_base", ['normal', 'radial'])
    optimizer = trial.suggest_categorical("optimizer", ['adam', 'nadam', 'sgd'])
    args.archi = trial.suggest_categorical("args.archi", 
            ['linear', 'conv', 'inception', 'bigception'])
    args.activation = trial.suggest_categorical("args.activation", 
        ['leaky_relu', 'tanh', 'relu'])
    args.last_layer = False #trial.suggest_categorical("args.last_layer", [True, False])
    args.pretrain = trial.suggest_categorical("args.pretrain", [0, 50])


    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    monitor = f"mse/val"
    trainer = pl.Trainer(
        gpus=[2],
        logger=True,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
    )

    hyperparams = {
        'bias' : True,
        'prior_loc' : prior_loc,
        'prior_scale' : prior_scale,
        'likelihood_scale' : likelihood_scale,
        'q_scale' : q_scale,
        'fit_context' : fit_context,
        'lr' : lr,
        'guide_base' : guide_base,
        'optimizer' : optimizer,
        'num_particles' : num_particles,
        'pretrain_file' : None,
        'last_layer': args.last_layer,
        'trial_id' : trial._trial_id,
        'device' : device,
    }


    if args.pretrain > 0:
        pre_net = DnnPretrainWrapper(data.win_length, data.n_features,
            archi = args.archi)
        pre_trainer = pl.Trainer(gpus=[2], max_epochs=args.pretrain, logger=False,
            enable_checkpointing=False)

        pretrain_dir = Path("lightning_logs",
            f'version_{trainer.logger.version}')
        pretrain_dir.mkdir(exist_ok=True, parents=True)
        hyperparams['pretrain_file'] = Path(pretrain_dir,
            f'pretrained_{args.pretrain}.pt').as_posix()
        pre_trainer.fit(pre_net, data)
        torch.save(pre_net.net.state_dict(), hyperparams['pretrain_file'])



    dnn = VIBnnWrapper(data.win_length, data.n_features, data.train_size,
        archi = args.archi, activation = args.activation, **hyperparams)
    
    
    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(dnn, datamodule=data)

    return trainer.callback_metrics[monitor].item()


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
    parser.add_argument('--model-name',
                    type=str,
                    default='dnn',
                    metavar='NAME',
                    help='Name of this specific run. (default: dnn)')
    parser.add_argument('--archi',
                    type=str,
                    default='linear',
                    metavar='ARCHI',
                    help='Which model to run. (default: linear')
    parser.add_argument('--pretrain',
                        type=int,
                        metavar='PRETRAIN',
                        default=0,
                        help='Pretrain the BNN weights for x epoch. (default: 0)')

    args = parser.parse_args()
    
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.NopPruner
    )
    study = optuna.create_study(
        direction="minimize",
        study_name="optuna_1", 
        pruner=pruner,
        storage="sqlite:///study.db",
        load_if_exists=True,
    )
    study.optimize(
        objective, 
        n_trials=500, 
        timeout=None,
        catch=(RuntimeError,),
    )
    #joblib.dump(study, "optuna2.pkl")
    df = study.trials_dataframe()

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))