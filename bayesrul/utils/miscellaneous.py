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
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
