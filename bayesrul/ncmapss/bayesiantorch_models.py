import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesrul.ncmapss.freq_models import NCMAPSSModel
from pytorch_lightning.utilities import rank_zero_only
from torch.functional import F


###############################################################################
##################################### BNN #####################################
###############################################################################


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
        num_predictions=50,
        lr=1e-3,
        weight_decay=1e-5,
        loss='mse'
    ):
        super().__init__(win_length, n_features, net, lr, weight_decay, loss)
        self.num_mc_samples_elbo = num_mc_samples_elbo
        self.num_predictions = num_predictions
        if const_bnn_prior_parameters is not None:
            dnn_to_bnn(self, const_bnn_prior_parameters)

        self.test_preds = {'preds': [], 'labels': [], 'std':[]}


    def _compute_loss(self, batch, phase):
        (x, y) = batch
        y = y.view(-1, 1)
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y)
        self.log(f"{self.loss}_loss/{phase}", loss)
        kl = get_kl_loss(self) / y.shape[0]
        loss = loss + kl
        self.log(f"kl_loss/{phase}", kl)
        self.log(f"total_loss/{phase}", loss)
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
        
        return {'loss': losses.mean(dim=0), 'std': losses.std(dim=0), 
                'label': batch[1]} 


    def test_epoch_end(self, outputs) -> None:
        for output in outputs:
            self.test_preds['preds'].extend(list(
                output['mean'].cpu().detach().numpy()))
            self.test_preds['labels'].extend(
                list(output['label'].cpu().detach().numpy()))
            self.test_preds['std'].extend(
                list(output['std'].cpu().detach().numpy()))