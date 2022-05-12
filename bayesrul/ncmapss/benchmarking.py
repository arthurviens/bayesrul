from email.mime import base
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.ncmapss.frequentist_models import NCMAPSSModel, get_checkpoint, TBLogger
from bayesrul.ncmapss.bayesian_models import NCMAPSSModelBnn
from bayesrul.utils.plotting import PredLogger

import torch
import pyro

import argparse


def complete_training_testing_freq(args):
    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    dnn = NCMAPSSModel(data.win_length, data.n_features, args.net)

    base_log_dir = Path(args.out_path, "frequentist", args.model_name)

    checkpoint_file = get_checkpoint(base_log_dir, version=None)

    logger = TBLogger(
        Path(base_log_dir),# ,f"lightning_logs/{args.net}"),
        default_hp_metric=False,
    )


    monitor = f"{dnn.loss}_loss/val"
    earlystopping_callback = EarlyStopping(monitor=monitor, patience=50)

    trainer = pl.Trainer(
        default_root_dir=base_log_dir,
        gpus=[0],
        max_epochs=1,
        log_every_n_steps=2,
        logger=logger,
        callbacks=[
            earlystopping_callback,
        ],
    )
    
    trainer.fit(dnn, data, ckpt_path=checkpoint_file)


    data = NCMAPSSDataModule(args.data_path, batch_size=1000)
    dnn = NCMAPSSModel.load_from_checkpoint(checkpoint_file)
    trainer = pl.Trainer(gpus=[0], log_every_n_steps=10, logger=logger, 
                        max_epochs=-1) # Silence warning
    
    trainer.test(dnn, data, verbose=False)
    predLog = PredLogger(base_log_dir)
    predLog.save(dnn.test_preds)



def complete_training_testing_tyxe(args):
    data = NCMAPSSDataModule(args.data_path, batch_size=10000)

    dnn = NCMAPSSModelBnn(data.win_length, data.n_features, data.train_size,
        archi = args.net, device=torch.device("cuda:0"))

    base_log_dir = Path(args.out_path, "bayesian", args.model_name)


    logger = TBLogger(
        Path(base_log_dir),# ,f"lightning_logs/{args.net}"),
        default_hp_metric=False,
    )

    monitor = f"{dnn.loss_name}/val"
    #checkpoint_callback = ModelCheckpoint(checkpoint_dir)
    earlystopping_callback = EarlyStopping(monitor=monitor, patience=5)

    trainer = pl.Trainer(
        default_root_dir=base_log_dir,
        gpus=[0], #
        max_epochs=1000,
        log_every_n_steps=2,
        logger=logger,
        callbacks=[
            earlystopping_callback,
        ],
    )

    trainer.fit(dnn, data)

    checkpoint_file = get_checkpoint(base_log_dir, version=None)
    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    dnn = NCMAPSSModelBnn.load_from_checkpoint(checkpoint_file, 
            map_location=torch.device("cuda:0"))

    trainer = pl.Trainer(
        gpus=[0], 
        log_every_n_steps=10, 
        logger=logger, 
        max_epochs=-1
        ) # Silence warning
    trainer.test(dnn, data, verbose=False)

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
    


    #complete_training_testing_freq(args)
    complete_training_testing_tyxe(args)