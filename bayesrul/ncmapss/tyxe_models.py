import os
import contextlib
from types import SimpleNamespace
from functools import partial
from pathlib import Path
from debugpy import configure
import pytorch_lightning as pl
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loops import Loop, FitLoop
import torch
import torch.nn as nn
from bayesrul.ncmapss.freq_models import get_checkpoint, TBLogger, Linear, Conv, weights_init
from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch.functional import F

import tyxe
from tyxe.bnn import _to
import pyro
import pyro.distributions as dist
import pyro.infer.autoguide as ag
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, MCMC
from tqdm import tqdm

args = SimpleNamespace(
    data_path="data/ncmapss/",
    out_path="results/ncmapss/",
    model_name="tyxe1",
    net="linear",
    lr=1e-4
)


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
        num_particles=1
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


        self.automatic_optimization = False
        self.configure_optimizers()
        closed_form_kl = False
        self.lr = lr
        self.mode = mode 
        self.num_particles = num_particles
        self.test_preds = {'preds': [], 'labels': [], 'stds': []}
        self.net.apply(weights_init)
        self.prior = tyxe.priors.IIDPrior(dist.Normal(
            torch.tensor(0.).to(self.device), 
            torch.tensor(1.).to(self.device)))
        self.obs_model = tyxe.likelihoods.HomoskedasticGaussian(dataset_size, scale=0.1)
        self.guide_builder = partial(tyxe.guides.AutoNormal, init_scale=0.01)
        self.bnn = tyxe.VariationalBNN(self.net.to(self.device), self.prior, self.obs_model, self.guide_builder)

        if closed_form_kl: self.loss = TraceMeanField_ELBO(num_particles)  
        else: self.loss = Trace_ELBO(num_particles)
        self.svi = SVI(self.bnn.model, self.bnn.guide, 
                        self.optimizer, loss=self.loss)


    def predict(self, data, num_pred):
        m, sd = self.bnn.predict(data, # GPU problem here 
            num_predictions=num_pred)
        return m, sd


    def test_step(self, batch, batch_idx):
        (x, y) = batch
        with self.fit_ctxt():
            m, sd = self.bnn.predict(x)
        mse = F.mse_loss(y, m.squeeze())
        self.log("mse_loss/test", mse)


    def validation_step(self, batch, batch_idx):
        (x, y) = batch
        with self.fit_ctxt():
            m, sd = self.predict(x, 1)
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
        self.optimizer = pyro.optim.Adam({"lr": 1e-3})



import matplotlib.pyplot as plt
if __name__ == "__main__":
    data = NCMAPSSDataModule(args.data_path, batch_size=1000)

    dnn = NCMAPSSModelBnn(data.win_length, data.n_features, data.train_size,
        net = args.net)

    base_log_dir = f"{args.out_path}/test/{args.model_name}/"

    checkpoint_dir = Path(base_log_dir, f"checkpoints/{args.net}")
    checkpoint_file = get_checkpoint(checkpoint_dir)

    logger = TBLogger(
        base_log_dir + f"lightning_logs/{args.net}",
        default_hp_metric=False,
    )

    trainer = pl.Trainer(
        #gpus=[0], # Does not work yet with TyXE
        max_epochs=100,
        log_every_n_steps=2,
        logger=logger,
    )

    trainer.fit(dnn, data)



    """
    net = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(data.n_features * data.win_length, 1)
        )
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1))
    likelihood = tyxe.likelihoods.HomoskedasticGaussian(87251, scale=0.1)
    inference = tyxe.guides.AutoNormal
    bnn = tyxe.VariationalBNN(net, prior, likelihood, inference)


    ##### Training loop
    data_loader = data.train_dataloader()
    optim = pyro.optim.Adam({"lr": 1e-3})
    num_particles = 1
    closed_form_kl = True
    epochs = 5
    device = torch.device('cpu')

    #bnn.fit(data_loader, optim, epochs, num_particles=num_particles, 
    #    closed_form_kl=closed_form_kl, device=device)

    #exit(-1)

    with tyxe.poutine.local_reparameterization():
        old_training_state = bnn.net.training
        bnn.net.train(True)
        if closed_form_kl: loss = TraceMeanField_ELBO(num_particles)  
        else: loss = Trace_ELBO(num_particles)
        svi = SVI(bnn.model, bnn.guide, optim, loss=loss)
    
        for i in tqdm(range(epochs)):
            elbo = 0
            num_batch=1
            for num_batch, (input_data, observation_data) in enumerate(
                tqdm(iter(data_loader), leave=False, position=1), 1):
                elbo += svi.step(tuple(_to(input_data, device)), tuple(_to(observation_data, device))[0])

    
    output_dir = 'results/ncmapss/test'
    Path(output_dir).mkdir(exist_ok=True)
    test_samples = 10
    
    if output_dir is not None:
        pyro.get_param_store().save(os.path.join(output_dir, "param_store.pt"))
        torch.save(bnn.state_dict(), os.path.join(output_dir, "state_dict.pt"))

        
        means = []
        stds = []
        for x, _ in iter(data.test_dataloader()):
            preds = bnn.predict(x.to(device), num_predictions=test_samples)
            means.append(preds[0])
            stds.append(preds[1])
        test_pred_means = torch.cat(means)
        test_pred_stds = torch.cat(stds)

        torch.save(test_pred_means.detach().cpu(), 
            os.path.join(output_dir, "test_predictions.pt"))
    """