import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.functional import F

from bayesrul.models.inception import InceptionModel, BigCeption
from bayesrul.models.linear import Linear
from bayesrul.models.conv import Conv
from bayesrul.utils.miscellaneous import weights_init, enable_dropout
from bayesrul.utils.metrics import MPIW, PICP, p_alphalamba

import numpy as np


class DnnWrapper(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        bias=True,
        archi="linear",
        out_size=1,
        lr=1e-3,
        weight_decay=1e-3,
        loss='mse',
        activation='relu',
        dropout=0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if archi == "linear":
            self.net = Linear(win_length, n_features, activation=activation,
                dropout=dropout, bias=bias, out_size=out_size)
        elif archi == "conv":
            self.net = Conv(win_length, n_features, activation=activation,
                dropout=dropout, bias=bias, out_size=out_size)
        elif archi == "inception":
            self.net = InceptionModel(win_length, n_features, out_size=out_size,
                dropout=dropout, activation=activation, bias=bias)
        elif archi == "bigception":
            self.net = BigCeption(n_features, activation=activation, 
                dropout=dropout, out_size=out_size, bias=bias)
        else:
            raise RuntimeError(f"Model architecture {archi} not implemented")

        if (loss == 'mse') or (loss == 'MSE'):
            self.criterion = F.mse_loss
            self.loss = 'mse'
        elif (loss == 'l1') or (loss == 'L1'):
            self.criterion = F.l1_loss
            self.loss = 'l1'
        else:
            raise RuntimeError(f"Loss {loss} not supported. Choose from"
                " ['mse', 'l1']")
                
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_preds = {'preds': [], 'labels': []}
        if dropout > 0: self.test_preds['stds'] = []
        self.dropout = dropout
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

    def _compute_loss(self, batch, phase, return_pred=False): 
        (x, y) = batch
        output = self.net(x)
        if output.shape[1] == 2:
            y_hat = output[:, 0]
            #scale = output[:, 1]
        else:
            y_hat = output.squeeze()
        
        loss = self.criterion(y_hat, y)        
        
        self.log(f"{self.loss}/{phase}", loss)
        if return_pred:
            return loss, y_hat
        else:
            return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        
        loss = self._compute_loss(batch, "val")
        if (self.dropout > 0) & self.current_epoch % 5 == 0: # MC-Dropout
            enable_dropout(self.net)
            preds = []
            losses = []
            for i in range(10):
                loss, pred = self._compute_loss(batch, "test", return_pred=True)
                preds.append(pred)
                losses.append(loss)
            loss = torch.stack(losses).mean()
            preds = torch.stack(preds)
            std = preds.std(axis=0)
            preds = preds.mean(axis=0)

            return {'loss': loss, 'label': batch[1], 'pred': pred, 'std': std} 
        else:
            return {'loss': loss}

    def validation_epoch_end(self, outputs) -> None:
        if (self.dropout > 0) & self.current_epoch % 5 == 0:
            preds = torch.tensor([])
            labels = torch.tensor([])
            stds = torch.tensor([])
            for output in outputs:
                preds = torch.cat([preds, output['pred'].cpu().detach()])
                labels = torch.cat([labels, output['label'].cpu().detach()])
                stds = torch.cat([stds, output['std'].cpu().detach()])

            mpiw = MPIW(
                preds, labels, normalized=True
            )
            picp = PICP(
                labels, preds, stds
            )
            alambda = p_alphalamba(labels, preds, stds)
            self.log(f"mpiw/val", mpiw)
            self.log(f"picp/val", picp)
            self.log(f"alambda/val", alambda)


    def test_step(self, batch, batch_idx):
        if self.dropout > 0:
            enable_dropout(self.net)

            preds = []
            losses = []
            for i in range(100):
                loss, pred = self._compute_loss(batch, "test", return_pred=True)
                preds.append(pred)
                losses.append(loss)
            
            loss = torch.stack(losses).mean()
            preds = torch.stack(preds)
            std = preds.std(axis=0)
            preds = preds.mean(axis=0)

            return {'loss': loss, 'label': batch[1], 'pred': pred, 'std': std} 
        else:
            loss, pred = self._compute_loss(batch, "test", return_pred=True)
            
            return {'loss': loss, 'label': batch[1], 'pred': pred} 

    def test_epoch_end(self, outputs):
        for output in outputs:

            preds = torch.tensor([])
            labels = torch.tensor([])
            stds = torch.tensor([])
            for output in outputs:
                preds = torch.cat([preds, output['pred'].cpu().detach()])
                labels = torch.cat([labels, output['label'].cpu().detach()])
                if self.dropout > 0:
                    stds = torch.cat([stds, output['std'].cpu().detach()])

        self.test_preds['preds'] = preds.numpy()
        self.test_preds['labels'] = labels.numpy()

        if self.dropout > 0:
            self.test_preds['stds'] = stds.numpy()
            mpiw = MPIW(
                stds, 
                labels, 
                normalized=True
            )
            picp = PICP(
                labels,
                preds,
                stds,
            )
            alambda = p_alphalamba(labels, preds, stds)
            self.log(f"mpiw/test", mpiw)
            self.log(f"picp/test", picp)
            self.log(f"alambda/test", alambda)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        """To initialize from checkpoint, without giving init args """
        parser = parent_parser.add_argument_group("DnnWrapper")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser

        
class DnnPretrainWrapper(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        bias=True,
        archi="linear",
        out_size=2,
        activation='relu',
        lr=1e-3,
        weight_decay=1e-3,
        loss='mse'
    ):
        super().__init__()
        self.save_hyperparameters()
        if archi == "linear":
            self.net = Linear(win_length, n_features, activation=activation,
                bias = bias, out_size=out_size)
        elif archi == "conv":
            self.net = Conv(win_length, n_features, activation=activation,
                bias = bias, out_size=out_size)
        elif archi == "inception":
            self.net = InceptionModel(win_length, n_features, out_size=out_size,
                activation=activation, bias=bias)
        elif archi == "bigception":
            self.net = BigCeption(win_length,n_features, out_size=out_size,
                activation=activation, bias=bias)
        else:
            raise RuntimeError(f"Model architecture {archi} not implemented")

        if (loss == 'mse') or (loss == 'MSE'):
            self.criterion = F.mse_loss
            self.loss = 'mse'
        elif (loss == 'l1') or (loss == 'L1'):
            self.criterion = F.l1_loss
            self.loss = 'l1'
        else:
            raise RuntimeError(f"Loss {loss} not supported. Choose from"
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
        parser = parent_parser.add_argument_group("DnnPretrainWrapper")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser



class DeepEnsembleWrapper(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        n_models, 
        bias=True,
        archi="linear",
        out_size=2,
        lr=1e-3,
        weight_decay=1e-3,
        activation='relu',
        dropout=0.1,
        device=torch.device('cuda:0'),
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if archi == "linear":
            self.nets = [Linear(win_length, n_features, activation=activation,
                dropout=dropout, bias=bias, out_size=out_size).to(device)
                for i in range(n_models)]
        elif archi == "conv":
            self.nets = [Conv(win_length, n_features, activation=activation,
                dropout=dropout, bias=bias, out_size=out_size).to(device)
                for i in range(n_models)]
        elif archi == "inception":
            self.nets = [InceptionModel(win_length, n_features, out_size=out_size,
                dropout=dropout, activation=activation, bias=bias).to(device)
                for i in range(n_models)]
        elif archi == "bigception":
            self.nets = [BigCeption(n_features, activation=activation, dropout=dropout, 
                out_size=out_size, bias=bias).to(device)
                for i in range(n_models)]
        else:
            raise RuntimeError(f"Model architecture {archi} not implemented")

        self.loss = 'nll_loss'
        self.criterion = F.gaussian_nll_loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_preds = {'preds': [], 'labels': []}
        if dropout > 0: self.test_preds['stds'] = []
        self.dropout = dropout
        for net in self.nets:
            net.apply(weights_init)

    def forward(self, x): # Non vérifié
        mus = []
        sigmas = []
        for net in self.nets:
            out = net(x)
            mus.append(out[:, 0])
            sigmas.append(out[:, 1])
        loc = torch.stack(mus)
        scale = torch.stack(sigmas)
        # Gaussian mixture formula
        var = (torch.square(scale) + torch.square(loc)).mean(0) - torch.square(loc.mean(0))
        return loc.mean(0), var

    def _compute_loss(self, batch, phase, return_pred=False): 
        (x, y) = batch
        loc, var = self.forward(x)
        loss = self.criterion(loc, y, var)        
        
        self.log(f"{self.loss}/{phase}", loss)
        if return_pred:
            return loss, loc, var
        else:
            return loss

    def training_step(self, batch, batch_idx):
        loss, loc, var = self._compute_loss(batch, "train", return_pred=True)
        mse = F.mse_loss(loc, batch[1])
        self.log("mse/train", mse)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loc, var = self._compute_loss(batch, "val", return_pred=True)
        mse = F.mse_loss(loc, batch[1])
        self.log("mse/val", mse)
        return {'loss': loss, 'label': batch[1], 'pred': loc, 'std': torch.sqrt(var)} 
        
    def validation_epoch_end(self, outputs) -> None:
        if (self.dropout > 0) & self.current_epoch % 5 == 0:
            preds = torch.tensor([])
            labels = torch.tensor([])
            stds = torch.tensor([])
            for output in outputs:
                preds = torch.cat([preds, output['pred'].cpu().detach()])
                labels = torch.cat([labels, output['label'].cpu().detach()])
                stds = torch.cat([stds, output['std'].cpu().detach()])

            mpiw = MPIW(
                preds, labels, normalized=True
            )
            picp = PICP(
                labels, preds, stds
            )
            alambda = p_alphalamba(labels, preds, stds)
            self.log(f"mpiw/val", mpiw)
            self.log(f"picp/val", picp)
            self.log(f"alambda/val", alambda)

    def test_step(self, batch, batch_idx):
        loss, loc, var = self._compute_loss(batch, "test", return_pred=True)
        mse = F.mse_loss(loc, batch[1])
        self.log("mse/test", mse)
        return {'loss': loss, 'label': batch[1], 'pred': loc, 'std': torch.sqrt(var)} 

    def test_epoch_end(self, outputs):
        for output in outputs:

            preds = torch.tensor([])
            labels = torch.tensor([])
            stds = torch.tensor([])
            for output in outputs:
                preds = torch.cat([preds, output['pred'].cpu().detach()])
                labels = torch.cat([labels, output['label'].cpu().detach()])
                if self.dropout > 0:
                    stds = torch.cat([stds, output['std'].cpu().detach()])

        self.test_preds['preds'] = preds.numpy()
        self.test_preds['labels'] = labels.numpy()

        if self.dropout > 0:
            self.test_preds['stds'] = stds.numpy()
            mpiw = MPIW(
                stds, 
                labels, 
                normalized=True
            )
            picp = PICP(
                labels,
                preds,
                stds,
            )
            alambda = p_alphalamba(labels, preds, stds)
            self.log(f"mpiw/test", mpiw)
            self.log(f"picp/test", picp)
            self.log(f"alambda/test", alambda)
        

    def configure_optimizers(self):
        params = []
        for net in self.nets:
            params.extend(list(net.parameters()))
        optimizer = torch.optim.Adam(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        """To initialize from checkpoint, without giving init args """
        parser = parent_parser.add_argument_group("DnnWrapper")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser