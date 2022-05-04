import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
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


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


# Model architectures based on:
# https://github.com/kkangshen/bayesian-deep-rul/blob/master/models/
# (Just model examples to be assessed and modified according to our needs)
class Linear(nn.Module):
    def __init__(self, win_length, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(win_length * n_features, 128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(64, 1)
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
            )
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
        weight_decay=1e-3,
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
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)

    def _compute_loss(self, batch, phase, return_pred=False): 
        (x, y) = batch
        y = y.view(-1, 1)
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y)
        self.log(f"{self.loss}_loss/{phase}", loss)
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
        parser = parent_parser.add_argument_group("NCMAPSSModel")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser

        

