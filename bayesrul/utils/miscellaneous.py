import os, glob
import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

# To get rid of the tensorboard epoch plot
class TBLogger(pl.loggers.TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)


def weights_init(m):
    """ Initializes weights of a nn.Module : xavier for conv
        and kaiming for linear
    
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


def get_checkpoint(path, version=None) -> None:
    try:
        path = os.path.join(os.getcwd(), path, 'lightning_logs')
        ls = sorted(os.listdir(path), reverse = True)
        d = os.path.join(path, ls[-1], "checkpoints")
        if os.path.isdir(d):
            checkpoint_file = sorted(
                glob.glob(os.path.join(d, "*.ckpt")), 
                key=os.path.getmtime, 
                reverse=True
            )
            return str(checkpoint_file[0]) if checkpoint_file else None
        return None
    except Exception as e:
        if e == FileNotFoundError:
            print("Could not find any checkpoint in {}".format(d))
        return None