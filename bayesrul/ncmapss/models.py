import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from pytorch_lightning.utilities import rank_zero_only
from torch.functional import F


def get_checkpoint(checkpoint_dir) -> None:
    if checkpoint_dir.is_dir():
        checkpoint_file = sorted(
            checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime, reverse=True
        )
        return str(checkpoint_file[0]) if checkpoint_file else None
    return None


# To get rid of the tensorboard epoch plot
class TBLogger(pl.loggers.TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)


# Model architectures based on:
# https://github.com/kkangshen/bayesian-deep-rul/blob/master/models/
# (Just model examples to be assessed and modified according to our needs)
class Linear(nn.Module):
    def __init__(self, win_length, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(win_length * n_features, 100),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.layers(x)


class Conv(nn.Module):
    def __init__(self, win_length, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 14)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            nn.Conv2d(8, 14, kernel_size=(2, 1)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            nn.Flatten(),
            nn.Linear(
                14 * int((((win_length - 4) / 2) - 1) / 2) * (n_features - 13), 1
            ),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.layers(x.unsqueeze(1))


class NCMAPSSModel(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        net="linear",
        lr=1e-3,
        weight_decay=1e-5,
        loss='mse'
    ):
        super().__init__()
        self.save_hyperparameters()
        if net == "linear":
            self.net = Linear(win_length, n_features)
        elif net == "conv":
            self.net = Conv(win_length, n_features)
        else:
            raise ValueError(f"Model architecture {net} not implemented")

        if (loss == 'mse') or (loss == 'MSE'):
            self.loss = F.mse_loss
            self.loss_name = 'mse'
        elif (loss == 'l1') or (loss == 'L1'):
            self.loss = F.l1_loss
            self.loss_name = 'l1'
        else:
            raise ValueError(f"Loss {loss} not supported. Choose from"
                " ['mse', 'l1']")
                
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.net(x)

    def _compute_loss(self, batch, phase): # More generic for other losses
        (x, y) = batch
        y = y.view(-1, 1)
        y_hat = self.net(x)
        loss = self.loss(y_hat, y)
        self.log(f"{self.loss_name}_loss/{phase}", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "test")
        return loss 


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NCMAPSSModel")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser


class NCMAPSSModelBnn(NCMAPSSModel):
    # Basic implementation example with bayesian-pytorch
    # TODO Likelihood models for data noise (see in TyXe tyxe.likelihoods module)
    def __init__(
        self,
        win_length,
        n_features,
        net="linear",
        const_bnn_prior_parameters=None,
        num_mc_samples_elbo=1,
        num_predictions=1,
        lr=1e-3,
        weight_decay=1e-5,
    ):
        super().__init__(win_length, n_features, net, lr, weight_decay)
        self.num_mc_samples_elbo = num_mc_samples_elbo
        self.num_predictions = num_predictions
        if const_bnn_prior_parameters is not None:
            dnn_to_bnn(self, const_bnn_prior_parameters)

    def _compute_loss(self, batch, phase):
        (x, y) = batch
        y = y.view(-1, 1)
        y_hat = self.net(x)
        mse_loss = F.mse_loss(y_hat, y)
        kl = get_kl_loss(self)
        loss = mse_loss + kl / len(batch)
        self.log(f"loss/{phase}", loss)
        self.log(f"loss_mse/{phase}", mse_loss)
        self.log(f"loss_kl/{phase}", kl / len(batch))
        return loss

    def training_step(self, batch, batch_idx):
        return torch.stack(
            [
                self._compute_loss(batch, "train")
                for _ in range(self.num_mc_samples_elbo)
            ]
        ).mean(dim=0)

    def validation_step(self, batch, batch_idx):
        return torch.stack(
            [self._compute_loss(batch, "val") for _ in range(self.num_predictions)]
        ).mean(dim=0)

    def test_step(self, batch, batch_idx):
        losses = torch.stack(
            [self._compute_loss(batch, "test") for _ in range(self.num_predictions)]
        )
        self.log(f'loss_std/test', losses.std(dim=0))
        # TODO STD, CI
        return losses.mean(dim=0)
