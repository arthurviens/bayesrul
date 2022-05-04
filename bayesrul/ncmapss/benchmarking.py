from cgi import test
from pathlib import Path
from types import SimpleNamespace
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.ncmapss.models import NCMAPSSModel, NCMAPSSModelBnn, get_checkpoint, TBLogger
from bayesrul.utils.plotting import PredLogger

import numpy as np
import pandas as pd

import argparse


def complete_training_testing_freq(args):
    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    dnn = NCMAPSSModel(data.win_length, data.n_features, args.net)

    base_log_dir = f"{args.out_path}/frequentist/{args.model_name}/"

    checkpoint_dir = Path(base_log_dir, f"checkpoints/{args.net}")
    checkpoint_file = get_checkpoint(checkpoint_dir)

    logger = TBLogger(
        base_log_dir + f"lightning_logs/{args.net}",
        default_hp_metric=False,
    )

    monitor = f"{dnn.loss}_loss/val"
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, monitor=monitor)
    earlystopping_callback = EarlyStopping(monitor=monitor, patience=50)

    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=1000,
        log_every_n_steps=2,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            earlystopping_callback,
        ],
    )
    trainer.fit(dnn, data, ckpt_path=checkpoint_file)

    data = NCMAPSSDataModule(args.data_path, batch_size=1000)
    dnn = NCMAPSSModel.load_from_checkpoint(get_checkpoint(checkpoint_dir))
    trainer = pl.Trainer(gpus=[0], log_every_n_steps=10, logger=logger, 
                        max_epochs=-1) # Silence warning
    
    trainer.test(dnn, data, verbose=False)
    predLog = PredLogger(base_log_dir)
    predLog.save(dnn.test_preds)





def complete_training_testing_bayes(args):
    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    dnn = NCMAPSSModelBnn(data.win_length, data.n_features, args.net,
            const_bnn_prior_parameters=args.const_bnn_prior_parameters)

    base_log_dir = f"{args.out_path}/bayesian/{args.model_name}/"

    checkpoint_dir = Path(base_log_dir, f"checkpoints/{args.net}")
    checkpoint_file = get_checkpoint(checkpoint_dir)

    logger = TBLogger(
        base_log_dir + f"lightning_logs/{args.net}",
        default_hp_metric=False,
    )

    monitor = f"{dnn.loss}_loss/val"
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, monitor=monitor)
    earlystopping_callback = EarlyStopping(monitor=monitor, patience=100)

    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=10000,
        log_every_n_steps=2,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            earlystopping_callback,
        ],
    )
    trainer.fit(dnn, data, ckpt_path=checkpoint_file)

    data = NCMAPSSDataModule(args.data_path, batch_size=1000)
    dnn = NCMAPSSModelBnn.load_from_checkpoint(get_checkpoint(checkpoint_dir))
    trainer = pl.Trainer(gpus=[0], log_every_n_steps=10, logger=logger, 
                        max_epochs=-1) # Silence warning

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
                    help='Name of this specific run. (default: dnn)')
    parser.add_argument('--net',
                    type=str,
                    default='linear',
                    metavar='NET',
                    help='Which model to run. (default: linear')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='learning rate (default: 1.0)')

    args = parser.parse_args()
    
    args.const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }


    complete_training_testing_freq(args)
    #complete_training_testing_bayes(args)