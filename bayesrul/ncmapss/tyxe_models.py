import os
import contextlib
from types import SimpleNamespace
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

"""
class Lite(LightningLite):
    def run(self, args, data, logger=None):
        optimizer = self.configure_optimizers()
        self.model = BNNWrapper(data.win_length, data.n_features, args.net)
        self.data = data
        self.logger = logger
        
        self.fit()

    def fit(self):
        dataloader = self.setup_dataloaders(self.data.train_dataloader())
        self.model.bnn.train(True)

        for epoch in tqdm(range(5)):
            for i, batch in enumerate(tqdm(dataloader, leave=False)):
                self.training_step(batch, i)
"""

class TyxeMainLoop(Loop):
    def __init__(self, model, dataloader, epochs):
        self.model = model 
        self.dataloader = dataloader 
        self.epochs = epochs
        self.epoch = 0
        self.fit_loop = TyxeFitLoop(self.model, self.dataloader)

    @property
    def done(self) -> bool:
        return self.epoch >= self.epochs

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self):
        pass

    def run(self):
        while not self.done:
            self.advance()
    
    def advance(self, *args, **kwargs) -> None:
        self.fit_loop.run(self.epoch)
        self.epoch += 1


class TyxeFitLoop(Loop):
    def __init__(self, model, dataloader):
        super().__init__()
        self.model = model
        self.dataloader = dataloader.train_dataloader()
        self.dataloader_iter = enumerate(self.dataloader)
        self.trainloader_size = len(self.dataloader)

    @property
    def done(self) -> bool:
        return False
    
    def reset(self) -> None:
        self.dataloader_iter = enumerate(self.dataloader)

    def advance(self, epoch) -> None:
        batch_idx, batch = next(self.dataloader_iter)
        self.model.training_step(batch, batch_idx)



class NCMAPSSModelBnn(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
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


        self.configure_optimizers()
        closed_form_kl = True
        self.lr = lr
        self.mode = mode 
        self.num_particles = num_particles
        self.test_preds = {'preds': [], 'labels': [], 'stds': []}
        self.net.apply(weights_init)
        self.prior = tyxe.priors.IIDPrior(dist.Normal(0, 1))
        self.likelihood = tyxe.likelihoods.HomoskedasticGaussian(87251, scale=0.1) # 87251
        self.inference = tyxe.guides.AutoNormal # ag.AutoMultivariateNormal 
        self.bnn = tyxe.VariationalBNN(self.net, self.prior, self.likelihood, self.inference)
        
        if closed_form_kl: self.loss = TraceMeanField_ELBO(num_particles)  
        else: self.loss = Trace_ELBO(num_particles)
        self.svi = SVI(self.bnn.model, self.bnn.guide, 
            self.optimizer, loss=self.loss)
        self.automatic_optimization = False


    def training_step(self, batch, batch_idx):
        (x, y) = batch
        with self.fit_ctxt():
            elbo = self.svi.step(x, y)
            print(elbo)


        exit(-1)
        self.log("loss/train", elbo)
        #return elbo

    def validation_step(self, batch, batch_idx):
        return 1

    def configure_optimizers(self):
        self.optimizer = pyro.optim.Adam({"lr": 1e-3})




if __name__ == "__main__":
    print("Hello, world!")
    data = NCMAPSSDataModule(args.data_path, batch_size=1000)
    #print(len(data.train_dataloader()))

    dnn = NCMAPSSModelBnn(data.win_length, data.n_features, args.net)

    base_log_dir = f"{args.out_path}/test/{args.model_name}/"

    checkpoint_dir = Path(base_log_dir, f"checkpoints/{args.net}")
    checkpoint_file = get_checkpoint(checkpoint_dir)

    logger = TBLogger(
        base_log_dir + f"lightning_logs/{args.net}",
        default_hp_metric=False,
    )

    trainer = pl.Trainer(
        #gpus=[0],
        max_epochs=100,
        log_every_n_steps=2,
        logger=logger,
    )

    #internal_fit_loop = trainer.fit_loop
    #outerloop = TyxeMainLoop(dnn, data, 5)
    #innerloop = TyxeFitLoop(dnn, data)
    #trainer.fit_loop = outerloop
    #trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(dnn, data)
    
    #Lite().run(args, data, logger=logger)




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