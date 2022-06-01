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
                dropout_freq=0, bias=True, typ="regression"):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'sigmoid':
            act = nn.Sigmoid
        elif activation == 'tanh':
            act = nn.Tanh
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU
        else:
            raise ValueError("Unknown activation")

        self.typ = typ
        if typ == "regression": out_size = 1
        elif typ == "classification": out_size = 10
        else: raise ValueError(f"Unknown value for typ : {typ}")

        if dropout_freq > 0 :
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(win_length * n_features, 256, bias=bias),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Linear(256, 128, bias=bias),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Linear(128, 128, bias=bias),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Linear(128, 64, bias=bias),
                nn.Dropout(p=dropout_freq),
                act(),
            )
        else:
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(win_length * n_features, 256, bias=bias),
                act(),
                nn.Linear(256, 128, bias=bias),
                act(),
                nn.Linear(128, 128, bias=bias),
                act(),
                nn.Linear(128, 32, bias=bias),
                act(),
            )
        self.last = nn.Linear(32, out_size)
        self.softmax = nn.Softmax(dim=1)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(),path)

    def load(self, path: str, map_location = torch.device('cuda:0')):
        state_dict = torch.load(path,map_location=map_location)
        self.load_state_dict(state_dict)

    def forward(self, x):
        if self.typ == "regression": 
            return self.last(self.layers(x.unsqueeze(1)))
        elif self.typ == "classification": 
            return self.softmax(self.last(self.layers(x.unsqueeze(1))))


class Conv(nn.Module):
    def __init__(self, win_length, n_features, activation='relu',
                dropout_freq=0, bias=True, typ='regression'):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'sigmoid':
            act = nn.Sigmoid
        elif activation == 'tanh':
            act = nn.Tanh
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU
        else:
            raise ValueError("Unknown activation")

        self.typ = typ
        if typ == "regression": out_size = 1
        elif typ == "classification": out_size = 10
        else: raise ValueError(f"Unknown value for typ : {typ}")

        if dropout_freq > 0: 
           self.layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(5, 9), bias=bias),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Conv2d(16, 32, kernel_size=(2, 10), bias=bias),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Conv2d(32, 64, kernel_size=(2, 1), bias=bias),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Flatten(),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(5, 9), bias=bias),
                act(),
                nn.Conv2d(16, 32, kernel_size=(2, 10), bias=bias),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Conv2d(32, 64, kernel_size=(2, 1), bias=bias),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Flatten(),
            )
        self.last = nn.Linear(
            64 * int((int((win_length - 5) / 2) - 1) / 2) * (n_features - 17), 
            out_size
        )
        self.softmax = nn.Softmax(dim=1)
            
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location = torch.device('cuda:0')):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

    def forward(self, x):
        if self.typ == "regression": 
            return self.last(self.layers(x.unsqueeze(1)))
        elif self.typ == "classification": 
            return self.softmax(self.last(self.layers(x.unsqueeze(1))))
        
class Conv2(nn.Module):
    def __init__(self, win_length, n_features, activation='relu',
                dropout_freq=0, bias=True, typ='regression'):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'sigmoid':
            act = nn.Sigmoid
        elif activation == 'tanh':
            act = nn.Tanh
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU
        else:
            raise ValueError("Unknown activation")

        self.typ = typ
        if typ == "regression": out_size = 1
        elif typ == "classification": out_size = 10
        else: raise ValueError(f"Unknown value for typ : {typ}")

        
        self.softmax = nn.LogSoftmax(-1)
        if dropout_freq > 0:
            self.layers = nn.Sequential(
                nn.Conv2d(1, 32, 4),
                nn.Dropout(p = dropout_freq),
                act(),
                nn.Conv2d(32, 32, 4),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(
                    int((win_length - 6) / 2) * int((n_features - 6) / 2) * 32, 
                    128
                ),
                nn.Dropout(p = dropout_freq),
                act()
            )
        else: 
            self.layers = nn.Sequential(
                nn.Conv2d(1, 32, 5, padding='same'),
                act(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 32, 5, padding='same'),
                act(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(
                    int(int((win_length + 4 - 4) / 2 + 4 - 4) / 2) 
                    * int(int((n_features + 4 - 4) / 2 + 4 - 4) / 2) * 32, 
                    1024
                ),
                act()
            )
        self.last = nn.Linear(1024, out_size)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location = torch.device('cuda:0')):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

    def forward(self, x):
        if self.typ == "regression": 
            return self.last(self.layers(x.unsqueeze(1)))
        elif self.typ == "classification": 
            return self.softmax(self.last(self.layers(x.unsqueeze(1))))
        


class NCMAPSSModel(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        bias=True,
        archi="linear",
        typ="regression",
        lr=1e-3,
        weight_decay=1e-3,
        loss='mse',
        activation='lrelu',
    ):
        super().__init__()
        self.save_hyperparameters()
        if archi == "linear":
            self.net = Linear(win_length, n_features, activation=activation,
                    dropout_freq=0.25, bias=bias, typ=typ)
        elif archi == "conv":
            self.net = Conv(win_length, n_features, activation=activation,
                    dropout_freq=0.25, bias=bias, typ=typ)
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
        bias=True,
        archi="linear",
        typ="regression",
        activation='relu',
        lr=1e-3,
        weight_decay=1e-3,
        loss='mse'
    ):
        super().__init__()
        self.save_hyperparameters()
        if archi == "linear":
            self.net = Linear(win_length, n_features, activation=activation,
                bias = bias, typ=typ)
        elif archi == "conv":
            self.net = Conv(win_length, n_features, activation=activation,
                bias = bias, typ=typ)
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