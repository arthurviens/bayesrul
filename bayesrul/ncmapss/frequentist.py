import os, glob

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.functional import F

from bayesrul.models.inception import InceptionModel, BigCeption
from bayesrul.models.linear import Linear
from bayesrul.models.conv import Conv
from bayesrul.utils.miscellaneous import weights_init


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
        activation='relu',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if archi == "linear":
            self.net = Linear(win_length, n_features, activation=activation,
                    dropout_freq=0.25, bias=bias, typ=typ)
        elif archi == "conv":
            self.net = Conv(win_length, n_features, activation=activation,
                    dropout_freq=0.25, bias=bias, typ=typ)
        elif archi == "inception":
            self.net = InceptionModel(activation=activation, bias=bias)
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
        output = self.net(x)
        try:
            y_hat = output[:, 0]
            sd = output[:, 1]
        except:
            y_hat = output
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
        elif archi == "inception":
            self.net = InceptionModel(activation=activation, bias = bias)
        elif archi == "bigception":
            self.net = BigCeption(activation=activation, bias=bias)
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
        y = y.view(-1, 1).to(torch.float32)
        output = self.net(x)
        try:
            y_hat = output[:, 0]
            sd = output[:, 1]
        except:
            y_hat = output
        #print(f"Output shape {output.shape}, then y_hat {y_hat.shape} then y {y.shape}")
        loss = self.criterion(y_hat, y.squeeze())
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