from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.ncmapss.frequentist import NCMAPSSModel
from bayesrul.ncmapss.frequentist import get_checkpoint, TBLogger

from bayesrul.utils.plotting import PredLogger

from bayesrul.inference.vi_bnn import VI_BNN


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
        data = NCMAPSSDataModule(args.data_path, batch_size=10000)
        module = VI_BNN(args, data)
        module.fit(2)
    else:
        complete_training_testing_freq(args)