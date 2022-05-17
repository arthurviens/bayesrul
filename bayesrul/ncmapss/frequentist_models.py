import os, glob

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only
from torch.functional import F


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


# To get rid of the tensorboard epoch plot
class TBLogger(pl.loggers.TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


# Model architectures based on:
# https://github.com/kkangshen/bayesian-deep-rul/blob/master/models/
# (Just model examples to be assessed and modified according to our needs)
class Linear(nn.Module):
    def __init__(self, win_length, n_features, activation='relu', 
                dropout_freq=0):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'sigmoid':
            act = nn.Sigmoid
        else:
            raise ValueError("Unknown activation")

        if dropout_freq > 0 :
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(win_length * n_features, 256),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Linear(256, 128),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Linear(128, 128),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Linear(128, 64),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Linear(64, 1)
            )
        else:
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(win_length * n_features, 256),
                act(),
                nn.Linear(256, 128),
                act(),
                nn.Linear(128, 128),
                act(),
                nn.Linear(128, 64),
                act(),
                nn.Linear(64, 1)
            )


    def forward(self, x):
        return self.layers(x)


class Conv(nn.Module):
    def __init__(self, win_length, n_features, activation='relu',
                dropout_freq=0):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'sigmoid':
            act = nn.Sigmoid
        else:
            raise ValueError("Unknown activation")
        if dropout_freq > 0: 
           self.layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(5, 9)),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Conv2d(16, 32, kernel_size=(2, 10)),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Conv2d(32, 64, kernel_size=(2, 1)),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Flatten(),
                nn.Linear(
                    64 * int((int((win_length - 5) / 2) - 1) / 2) * (n_features - 17), 1
                )
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(5, 9)),
                act(),
                nn.Conv2d(16, 32, kernel_size=(2, 10)),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Conv2d(32, 64, kernel_size=(2, 1)),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Flatten(),
                nn.Linear(
                    64 * int((int((win_length - 5) / 2) - 1) / 2) * (n_features - 17), 1
                )
            )
            
    def forward(self, x):
        return self.layers(x.unsqueeze(1))


class NCMAPSSModel(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        archi="linear",
        lr=1e-3,
        weight_decay=1e-3,
        loss='mse'
    ):
        super().__init__()
        self.save_hyperparameters()
        if archi == "linear":
            self.net = Linear(win_length, n_features, dropout_freq=0.25)
        elif archi == "conv":
            self.net = Conv(win_length, n_features, dropout_freq=0.25)
        else:
            raise ValueError(f"Model architecture {archi} not implemented")

        if (loss == 'mse') or (loss == 'MSE'):
            self.criterion = F.mse_loss
            self.loss = 'mse'
        elif (loss == 'l1') or (loss == 'L1'):
            self.criterion = F.l1_loss
            self.loss = 'l1'
        else:
            raise ValueError(f"Loss {loss} not supported. Choose from"
                " ['mse', 'l1']")
                
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_preds = {'preds': [], 'labels': []}
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

    def _compute_loss(self, batch, phase, return_pred=False): 
        (x, y) = batch
        y = y.view(-1, 1)
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y)
        self.log(f"{self.loss}/{phase}", loss)
        if return_pred:
            return loss, y_hat
        else:
            return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        loss, pred = self._compute_loss(batch, "test", return_pred=True)
                
        return {'loss': loss, 'label': batch[1], 'pred': pred} 

    def test_epoch_end(self, outputs):
        for output in outputs:
            self.test_preds['preds'].extend(list(
                output['pred'].flatten().cpu().detach().numpy()))
            #stds.extend(list(output['std'].cpu().detach().numpy()))
            self.test_preds['labels'].extend(list(
                output['label'].cpu().detach().numpy()))
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        """To initialize from checkpoint, without giving init args """
        parser = parent_parser.add_argument_group("NCMAPSSModel")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser

        
class NCMAPSSPretrain(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        archi="linear",
        lr=1e-3,
        weight_decay=1e-3,
        loss='mse'
    ):
        super().__init__()
        self.save_hyperparameters()
        if archi == "linear":
            self.net = Linear(win_length, n_features)
        elif archi == "conv":
            self.net = Conv(win_length, n_features)
        else:
            raise ValueError(f"Model architecture {archi} not implemented")

        if (loss == 'mse') or (loss == 'MSE'):
            self.criterion = F.mse_loss
            self.loss = 'mse'
        elif (loss == 'l1') or (loss == 'L1'):
            self.criterion = F.l1_loss
            self.loss = 'l1'
        else:
            raise ValueError(f"Loss {loss} not supported. Choose from"
                " ['mse', 'l1']")
                
        self.lr = lr
        self.weight_decay = weight_decay
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

    def _compute_loss(self, batch, phase, return_pred=False): 
        (x, y) = batch
        y = y.view(-1, 1)
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y)
        if return_pred:
            return loss, y_hat
        else:
            return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        """To initialize from checkpoint, without giving init args """
        parser = parent_parser.add_argument_group("NCMAPSSPretrain")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser