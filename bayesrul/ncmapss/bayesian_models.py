import os
import contextlib
from types import SimpleNamespace
from functools import partial
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loops import Loop, FitLoop
import torch
import torch.nn as nn
from bayesrul.ncmapss.frequentist_models import get_checkpoint, TBLogger, Linear, Conv, weights_init
from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from torch.functional import F

import tyxe
import pyro
import pyro.distributions as dist
import pyro.infer.autoguide as ag
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, MCMC
from tqdm import tqdm

from pytorch_lightning.plugins import CheckpointIO


args = SimpleNamespace(
    data_path="data/ncmapss/",
    out_path="results/ncmapss/",
    model_name="tyxe1",
    net="linear",
    lr=1e-4
)


def remove_dict_entry_startswith(dictionary, string):
    n = len(string)
    for key in dictionary:
        if string == key[:n]:
            print(f"To remove {key}")
            dict2 = dictionary.copy()
            dict2.pop(key)
            dictionary = dict2
    print(dictionary.keys())
    return dictionary



class NCMAPSSModelBnn(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        dataset_size,
        net="linear",
        mode="vi",
        fit_context="lrt",
        lr=1e-3,
        num_particles=1,
        device=torch.device('cuda:0')
    ):
        super().__init__()
        
        if net == "linear":
            self.net = Linear(win_length, n_features)
        elif net == "conv":
            self.net = Conv(win_length, n_features)
        else:
            raise ValueError(f"Model architecture {net} not implemented")

        if fit_context == 'lrt':
            self.fit_ctxt = tyxe.poutine.local_reparameterization
        elif fit_context == 'flipout':
            self.fit_ctxt = tyxe.poutine.flipout
        else:
            self.fit_ctxt = contextlib.nullcontext

        self.dummy = nn.Linear(1,1)
        #self.automatic_optimization = False
        self.compute_freqloss = False # True does not work yet on GPU
        self.configure_optimizers()
        closed_form_kl = False
        self.lr = lr
        self.mode = mode 
        self.num_particles = num_particles
        self.test_preds = {'preds': [], 'labels': [], 'stds': []}
        self.net.apply(weights_init)
        self.prior = tyxe.priors.IIDPrior(
            dist.Normal(
                torch.tensor(0.0, device=device), 
                torch.tensor(1.0, device=device)
            )
        )
        self.likelihood = tyxe.likelihoods.HomoskedasticGaussian(
            dataset_size, scale=0.5
        )
        self.guide = partial(tyxe.guides.AutoNormal, init_scale=0.5)
        self.bnn = tyxe.VariationalBNN(self.net, self.prior, self.likelihood, self.guide)
        self.loss_name = "elbo"
        self.svi = SVI(
            self.bnn.model,
            self.bnn.guide,
            pyro.optim.Adam({"lr": lr}),
            loss=(
                TraceMeanField_ELBO(num_particles)
                if closed_form_kl
                else Trace_ELBO(num_particles)
            ),
        )

        self.test_preds = {'labels': [], 'preds': [], 'stds': []}


    def test_step(self, batch, batch_idx):
        (x, y) = batch
        with self.fit_ctxt():
            m, sd = self.bnn.predict(x, num_predictions=100)
        
        mse = F.mse_loss(y, m.squeeze())
        self.log("mse_loss/test", mse)

        return {"loss": mse, "label": batch[1], "pred": m.squeeze(), "std": sd}


    def test_epoch_end(self, outputs):
        for output in outputs:
            self.test_preds['preds'].extend(list(
                output['pred'].flatten().cpu().detach().numpy()))
            self.test_preds['labels'].extend(list(
                output['label'].cpu().detach().numpy()))
            self.test_preds['stds'].extend(list(
                output['std'].flatten().cpu().detach().numpy()))


    def validation_step(self, batch, batch_idx):
        (x, y) = batch
        with self.fit_ctxt():
            elbo = self.svi.evaluate_loss(x, y)
        self.log('elbo/val', elbo)

        with self.fit_ctxt():
            m, sd = self.bnn.predict(x)
        mse = F.mse_loss(y, m.squeeze())
        self.log("mse_loss/val", mse)


    def training_step(self, batch, batch_idx):
        (x, y) = batch
        
        with self.fit_ctxt():
            elbo = self.svi.step(x, y)
            m, sd = self.bnn.predict(x)
    
        mse = F.mse_loss(y, m.squeeze()).item()
        self.log("mse_loss/train", mse)
        self.log("elbo/train", elbo)


    def configure_optimizers(self):
        return torch.optim.Adam(self.dummy.parameters(), lr=0.1)


    def on_save_checkpoint(self, checkpoint):
        checkpoint["param_store"] = pyro.get_param_store().get_state()

    def on_load_checkpoint(self, checkpoint):
        pyro.get_param_store().set_state(checkpoint["param_store"])
        checkpoint['state_dict'] = remove_dict_entry_startswith(checkpoint['state_dict'], 'bnn')




