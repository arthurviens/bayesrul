from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torchinfo import summary

from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.ncmapss.frequentist_models import NCMAPSSModel, NCMAPSSPretrain
from bayesrul.ncmapss.frequentist_models import get_checkpoint, TBLogger
from bayesrul.ncmapss.bayesian_models import NCMAPSSBnn
from bayesrul.utils.plotting import PredLogger

import torch

import argparse


def complete_training_testing_freq(args):
    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    dnn = NCMAPSSModel(data.win_length, data.n_features, archi = args.archi)

    base_log_dir = Path(args.out_path, "frequentist", args.model_name)

    checkpoint_file = get_checkpoint(base_log_dir, version=None)

    logger = TBLogger(
        Path(base_log_dir),
        default_hp_metric=False,
    )


    monitor = f"{dnn.loss}/val"
    earlystopping_callback = EarlyStopping(monitor=monitor, patience=50)

    trainer = pl.Trainer(
        default_root_dir=base_log_dir,
        gpus=[1],
        max_epochs=1000,
        log_every_n_steps=2,
        logger=logger,
        callbacks=[
            earlystopping_callback,
        ],
    )
    
    trainer.fit(dnn, data, ckpt_path=checkpoint_file)


    checkpoint_file = get_checkpoint(base_log_dir, version=None)

    data = NCMAPSSDataModule(args.data_path, batch_size=1000)
    dnn = NCMAPSSModel.load_from_checkpoint(checkpoint_file)
    trainer = pl.Trainer(gpus=[0], log_every_n_steps=10, logger=logger, 
                        max_epochs=-1) # Silence warning
    
    trainer.test(dnn, data, verbose=False)
    predLog = PredLogger(base_log_dir)
    predLog.save(dnn.test_preds)



def complete_training_testing_tyxe(args, hyperparams=None, GPU = 1):
    if hyperparams is None:
        hyperparams = {
            'bias' : True,
            'prior_loc' : 0,
            'prior_scale' : 0.5,
            'likelihood_scale' : 10,
            'q_scale' : .001,
            'mode' : 'vi',
            'fit_context' : 'lrt',
            'num_particles' : 1,
            'optimizer': 'adam',
            'lr' : 1e-3,
            'last_layer': args.last_layer,
            'pretrain_file' : None,
        }

    print(args)
    print(hyperparams)
    if args.guide == "radial": hyperparams["fit_context"] = 'null'

    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    base_log_dir = Path(args.out_path, "bayesian", args.model_name)
    
    logger = TBLogger(
        Path(base_log_dir),# ,f"lightning_logs/{args.net}"),
        default_hp_metric=False,
    )

    monitor = f"mse/val"
    #earlystopping_callback = EarlyStopping(monitor=monitor, patience=50)
    trainer = pl.Trainer(
        default_root_dir=base_log_dir,
        gpus=[GPU], 
        max_epochs=150,
        log_every_n_steps=2,
        logger=logger,
        #callbacks=[earlystopping_callback],
    )

    checkpoint_file = get_checkpoint(base_log_dir, version=None)
    if args.pretrain > 0 and checkpoint_file:
        raise ValueError("Can not pretrain and resume from checkpoint")

    if args.pretrain > 0 and (not checkpoint_file):
        pre_net = NCMAPSSPretrain(data.win_length, data.n_features,
            archi = args.archi, bias=hyperparams['bias'])
        pre_trainer = pl.Trainer(gpus=[GPU], max_epochs=args.pretrain, logger=False,
            checkpoint_callback=False)

        pretrain_dir = Path(base_log_dir, "lightning_logs",
            f'version_{trainer.logger.version}')
        pretrain_dir.mkdir(exist_ok=True, parents=True)
        hyperparams['pretrain_file'] = Path(pretrain_dir,
            f'pretrained_{args.pretrain}.pt').as_posix()
        pre_trainer.fit(pre_net, data)
        base_log_dir.mkdir(exist_ok=True)
        torch.save(pre_net.net.state_dict(), hyperparams['pretrain_file'])
        
    if checkpoint_file:
        dnn = NCMAPSSBnn.load_from_checkpoint(checkpoint_file,
            map_location=torch.device(f"cuda:{GPU}"))
    else:
        dnn = NCMAPSSBnn(data.win_length, data.n_features, data.train_size,
            archi = args.archi, device=torch.device(f"cuda:{GPU}"), 
            guide_base = args.guide, **hyperparams)



    trainer.fit(dnn, data)

    tester = pl.Trainer(
        gpus=[GPU], 
        log_every_n_steps=10, 
        logger=logger, 
        max_epochs=-1 # Silence warning
        ) 
    tester.test(dnn, data, verbose=False)

    predLog = PredLogger(base_log_dir)
    predLog.save(dnn.test_preds)



if __name__ == "__main__":
    # Launch from root directory : python -m bayesrul.ncmapss.benchmarking
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
                    help='Name of this specific run. (default: dnn)',
                    required=True)
    parser.add_argument('--archi',
                    type=str,
                    default='linear',
                    metavar='ARCHI',
                    help='Which model to run. (default: linear)')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--pretrain',
                        type=int,
                        metavar='PRETRAIN',
                        default=0,
                        help='Pretrain the BNN weights for x epoch. (default: 0)')
    parser.add_argument('--bayesian',
                        action='store_true',
                        default=False,
                        help='Wether to train a bayesian model (default: False)')
    parser.add_argument('--last-layer',
                        action='store_true',
                        default=False,
                        help='Having only the last layer as Bayesian (default: False)')
    parser.add_argument('--guide',
                    type=str,
                    default='normal',
                    metavar='GUIDE',
                    help='Normal or Radial Autoguide. (default: normal)')
 

    args = parser.parse_args()
    

    if args.bayesian:
        complete_training_testing_tyxe(args)
    else:
        complete_training_testing_freq(args)