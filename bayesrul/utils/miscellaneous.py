import os, glob
import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



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


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    assert isinstance(m, torch.nn.Module), f"{type(m)} is not a nn.Module"
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)