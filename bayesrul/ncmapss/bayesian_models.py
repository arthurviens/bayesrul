import contextlib
from types import SimpleNamespace
from functools import partial
import pytorch_lightning as pl
import torch
import torch.nn as nn
from bayesrul.ncmapss.frequentist_models import Linear, Conv, weights_init
from torch.functional import F


import tyxe
import pyro
import pyro.distributions as dist
import pyro.nn as pynn
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, MCMC



def remove_dict_entry_startswith(dictionary, string):
    """Used to remove entries with 'bnn' in checkpoint state dict"""
    n = len(string)
    for key in dictionary:
        if string == key[:n]:
            dict2 = dictionary.copy()
            dict2.pop(key)
            dictionary = dict2
    return dictionary


class NCMAPSSModelBnn(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        dataset_size,
        prior_loc=0.,
        prior_scale=1.,
        likelihood_scale=0.5,
        vardist_scale=0.5,
        archi="linear",
        mode="vi",
        fit_context="lrt",
        lr=1e-3,
        num_particles=1,
        device=torch.device('cuda:0'),
        pretrain_file=None,
        pretrain_init_scale=0.01,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if archi == "linear":
            self.net = Linear(win_length, n_features).to(device=device)
        elif archi == "conv":
            self.net = Conv(win_length, n_features).to(device=device)
        else:
            raise ValueError(f"Model architecture {archi} not implemented")

        if pretrain_file is not None:
            sd = torch.load(pretrain_file, map_location=device)
            self.net.load_state_dict(sd)

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
        self.loss_name = "elbo"
        self.test_preds = {'preds': [], 'labels': [], 'stds': []}
        self.net.apply(weights_init)
        self.prior = tyxe.priors.IIDPrior(
            dist.Normal(
                torch.tensor(float(prior_loc), device=device), 
                torch.tensor(float(prior_scale), device=device)
            )
        )
        self.likelihood = tyxe.likelihoods.HomoskedasticGaussian(
            dataset_size, scale=likelihood_scale
        )
        if pretrain_file is not None:
            self.guide = partial(
                tyxe.guides.AutoNormal,
                init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(self.net),
                init_scale=pretrain_init_scale
                )
        else:
            self.guide = partial(tyxe.guides.AutoNormal, init_scale=vardist_scale)
        self.bnn = tyxe.VariationalBNN(
            self.net, 
            self.prior, 
            self.likelihood, 
            self.guide, 
            name="bnn"
            )
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
        
        # As we do not use PyTorch Optimizers, it is needed in order to Pytorch
        # Lightning to know that we are training a model, and initiate routines
        # like checkpointing etc.
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.\
                                    optim_step_progress.increment_ready()
        with self.fit_ctxt():
            elbo = self.svi.step(x, y)
            m, sd = self.bnn.predict(x)
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.\
                                    optim_step_progress.increment_completed()
        
        mse = F.mse_loss(y, m.squeeze()).item()
        self.log("mse_loss/train", mse)
        self.log("elbo/train", elbo)


    def configure_optimizers(self):
        """Because we use Pyro's SVI optimizer that works differently"""
        return None


    def on_save_checkpoint(self, checkpoint):
        """Saving Pyro's param_store for the bnn's parameters"""
        checkpoint["param_store"] = pyro.get_param_store().get_state()
        

    def on_load_checkpoint(self, checkpoint):
        pyro.get_param_store().set_state(checkpoint["param_store"])
        checkpoint['state_dict'] = remove_dict_entry_startswith(
            checkpoint['state_dict'], 'bnn')


    @staticmethod
    def add_model_specific_args(parent_parser):
        """To initialize from checkpoint, without giving init args """
        parser = parent_parser.add_argument_group("NCMAPSSModelBnn")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser


