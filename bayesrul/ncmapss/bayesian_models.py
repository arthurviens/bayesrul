import contextlib
from types import SimpleNamespace
from functools import partial
import pytorch_lightning as pl
import torch
import torch.nn as nn
from bayesrul.ncmapss.frequentist_models import Linear, Conv, Conv2, weights_init
from bayesrul.ncmapss.inception import InceptionModel
from tyxe.bnn import VariationalBNN
from bayesrul.ncmapss.radial import AutoRadial
from bayesrul.ncmapss.custom_elbos import CustomTrace_ELBO
from torch.functional import F


import tyxe
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
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

def inverse_softplus(t):
    return t.expm1().log()

def custom_elbo(model, guide, *args, **kwargs):
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    model_trace = poutine.trace(poutine.replay(model, guide_trace)).get_trace(
        *args, **kwargs
    )
    
    try:
        obs = args[1].view(-1, 1)
    except IndexError:
        return 0.
    #for name, model_site in model_trace.nodes.items():
    #    print(f"---------------> NODE {name}")
    #    print(model_site)
    pred, var = model_trace.nodes["likelihood.data_plate"]["value"].chunk(2, dim=-1)
    nll_loss = nn.GaussianNLLLoss(reduction="sum")
    nll = nll_loss(pred, obs, var)
    return nll


class NCMAPSSBnn(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        dataset_size,
        bias = True,
        prior_loc=0.,
        prior_scale=10,
        prior='gaussian',
        likelihood_scale=3,
        q_scale=0.01,
        archi="inception",
        activation="relu",
        mode="vi",
        fit_context="flipout",
        guide_base="normal",
        optimizer='nadam',
        lr=1e-3,
        num_particles=1,
        device=torch.device('cuda:0'),
        pretrain_file=None,
        last_layer=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        closed_form_kl = True
        
        if archi == "linear":
            self.net = Linear(win_length, n_features, activation=activation,
                bias=bias, typ='regression').to(device=device)
        elif archi == "conv":
            self.net = Conv(win_length, n_features, activation=activation, 
                bias=bias, typ='regression').to(device=device)
        elif archi == "inception":
            self.net = InceptionModel(activation=activation, 
                bias=bias).to(device=device)
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

        if guide_base == 'normal':
            guide_base = tyxe.guides.AutoNormal
        elif guide_base == 'radial':
            guide_base = AutoRadial
            closed_form_kl = False
            print("Using Radial Guide")
        else: 
            raise ValueError("Guide unknown. Choose from 'normal', 'radial'.")
        

        self.dataset_size = dataset_size
        self.lr = lr
        self.mode = mode 
        self.num_particles = num_particles
        self.loss_name = "elbo"
        """self.loss = (
            TraceMeanField_ELBO(num_particles)
            if closed_form_kl
            else Trace_ELBO(num_particles)
        )"""
        self.loss = CustomTrace_ELBO(num_particles)
        self.opt_name = optimizer
        self.configure_optimizers()
        self.test_preds = {'preds': [], 'labels': [], 'stds': []}
        self.net.apply(weights_init)
        prior_kwargs = {'hide_all': True}
        
        if prior == 'gaussian':
            self.prior = tyxe.priors.IIDPrior(
                dist.Normal(
                    torch.tensor(float(prior_loc), device=device), 
                    torch.tensor(float(prior_scale), device=device),
                ), **prior_kwargs
            )
        elif prior == 'laplace':
            self.prior = tyxe.priors.IIDPrior(
                dist.Laplace(
                    torch.tensor(float(prior_loc), device=device), 
                    torch.tensor(float(prior_scale), device=device),
                ), **prior_kwargs
            )
            closed_form_kl = False
        else: 
            raise ValueError("Guide unknown. Choose from 'normal', 'radial'.")
        
        self.likelihood = tyxe.likelihoods.HeteroskedasticGaussian(
            dataset_size, #scale=likelihood_scale
        )
        if pretrain_file is not None:
            print("Initializing weight distributions from pretrained net")
            self.net.load(pretrain_file, map_location=device)
            
            if not last_layer:
                self.guide = partial(
                    guide_base,
                    init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(self.net),
                    init_scale=q_scale
                )
            else:
                print("Last_layer training only")
                for module in self.net.modules():
                    if module is not self.net.last: # -> last layer !
                        for param_name, param in list(module.named_parameters(recurse=False)):
                            delattr(module, param_name)
                            module.register_buffer(param_name, param.detach().data)
                
                self.guide = partial(guide_base,
                    init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(self.net), 
                    init_scale=q_scale)

        else:
            if last_layer > 0:
                raise ValueError("No pretrain file but last_layer True")
            
            self.guide = partial(guide_base, init_scale=q_scale)
        self.guide = None
        
        self.bnn = VariationalBNN(
            self.net, 
            self.prior, 
            self.likelihood, 
            self.guide,
            )
        self.svi = SVI(
            pyro.poutine.scale(self.bnn.model, scale=1./(dataset_size * win_length * n_features)),
            pyro.poutine.scale(self.bnn.guide, scale=1./(dataset_size * win_length * n_features)),
            self.optimizer,
            self.loss
        )
        self.test_preds = {'labels': [], 'preds': [], 'stds': []}



    def test_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1].squeeze()
        with self.fit_ctxt():
            output = self.bnn.predict(x, num_predictions=5)
            if isinstance(output, tuple):
                m, sd = output
            else:
                m, sd = output[:, 0], output[:, 1]
        
        mse = F.mse_loss(y.squeeze(), m.squeeze())
        self.log("mse/test", mse)

        try:
            return {"loss": mse, "label": batch[1], "pred": m.squeeze(), "std": sd}
        except NameError:
            return {"loss": mse, "label": batch[1], "pred": m.squeeze()}

    def test_epoch_end(self, outputs):
        for output in outputs:
            self.test_preds['preds'].extend(list(
                output['pred'].flatten().cpu().detach().numpy()))
            self.test_preds['labels'].extend(list(
                output['label'].cpu().detach().numpy()))
            self.test_preds['stds'].extend(list(
                output['std'].flatten().cpu().detach().numpy()))


    def validation_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1].squeeze()
        
        # To compute only the KL part of the loss (no obs = no likelihood)
        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
        self.svi_no_obs = SVI(self.bnn_no_obs, self.bnn.guide,
            self.optimizer, self.loss)

        with self.fit_ctxt():
            elbo = self.svi.evaluate_loss(x, y)
            output = self.bnn.predict(x, num_predictions=5)
            if isinstance(output, tuple):
                m, sd = output
            else:
                m, sd = output[:, 0], output[:, 1]
            kl = self.svi_no_obs.evaluate_loss(x)
        
        mse = F.mse_loss(y.squeeze(), m.squeeze())

        self.log('elbo/val', elbo)
        self.log("mse/val", mse)
        self.log('kl/val', kl)
        self.log('likelihood/val', elbo - kl)
        #return {'loss' : mse}


    def training_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1]#.squeeze()
        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
        self.svi_no_obs = SVI(self.bnn_no_obs, self.bnn.guide,
            self.optimizer, self.loss)

        # As we do not use PyTorch Optimizers, it is needed in order to Pytorch
        # Lightning to know that we are training a model, and initiate routines
        # like checkpointing etc.
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.\
                                    optim_step_progress.increment_ready()
        with self.fit_ctxt():
            elbo = self.svi.step(x, y.squeeze())
            output = self.bnn.predict(x, num_predictions=1)
            if isinstance(output, tuple):
                m, sd = output
            else:
                m, sd = output[:, 0], output[:, 1]
            kl = self.svi_no_obs.evaluate_loss(x)
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.\
                                    optim_step_progress.increment_completed()
        
        mse = F.mse_loss(y.squeeze(), m.squeeze()).item()
        self.log("mse/train", mse)
        self.log("elbo/train", elbo)
        self.log('kl/train', kl)
        self.log('likelihood/train', elbo - kl)
        #return {'loss' : mse}


    def configure_optimizers(self):
        """Because we use Pyro's SVI optimizer that works differently"""
        if self.opt_name == "radam":
            self.optimizer = pyro.optim.RAdam({"lr": self.lr})
        elif self.opt_name == "sgd":
            self.optimizer = pyro.optim.SGD({"lr": self.lr})
        elif self.opt_name == "adagrad": 
            self.optimizer = pyro.optim.Adagrad({"lr": self.lr})
        elif self.opt_name == "adam":
            self.optimizer = pyro.optim.Adam({"lr": self.lr, "betas": (0.95, 0.999)})
        elif self.opt_name == "nadam":
            self.optimizer = pyro.optim.NAdam({"lr": self.lr})
        else:
            raise ValueError("Unknown optimizer")
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

"""
class BnnClassifier(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        dataset_size,
        bias = True,
        prior_loc=0.1,
        prior_scale=0.1,
        q_scale=0.001,
        archi="linear",
        activation="relu",
        mode="vi",
        fit_context="flipout",
        guide_base="normal",
        lr=1e-3,
        num_particles=1,
        device=torch.device('cuda:0'),
        pretrain_file=None,
        last_layer=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        closed_form_kl = True
        
        if archi == "linear":
            self.net = Linear(win_length, n_features, activation=activation,
                bias=bias, typ='classification').to(device=device)
        elif archi == "conv":
            self.net = Conv(win_length, n_features, activation=activation, 
                bias=bias, typ='classification').to(device=device)
        elif archi == "conv2":
            self.net = Conv2(win_length, n_features, activation=activation, 
                bias=bias, typ='classification').to(device=device)
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

        if guide_base == 'normal':
            guide_base = tyxe.guides.AutoNormal
        elif guide_base == 'radial':
            guide_base = AutoRadial
            closed_form_kl = False
            print("Using Radial Guide")
        else: 
            raise ValueError("Guide unknown. Choose from 'normal', 'radial'.")
        
        #    self.fit_ctxt().__enter__()
        self.lr = lr
        self.mode = mode 
        self.num_particles = num_particles
        self.loss_name = "elbo"
        self.loss = (
            TraceMeanField_ELBO(num_particles)
            if closed_form_kl
            else Trace_ELBO(num_particles)
        )
        self.configure_optimizers()
        self.test_preds = {'preds': [], 'labels': [], 'stds': []}
        self.net.apply(weights_init)
        prior_kwargs = {}#{'hide_all': True}
        self.prior = tyxe.priors.IIDPrior(
            dist.Normal(
                torch.tensor(float(prior_loc), device=device), 
                torch.tensor(float(prior_scale), device=device),
            ), **prior_kwargs
        )
        self.likelihood = tyxe.likelihoods.Categorical(-1)
        self.likelihood.dataset_size = dataset_size
        if pretrain_file is not None:
            print("Initializing weight distributions from pretrained net")
            self.net.load(pretrain_file, map_location=device)
            
            if not last_layer:
                self.guide = partial(
                    guide_base,
                    init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(self.net),
                    init_scale=q_scale
                )
            else:
                print("Last_layer training only")
                for module in self.net.modules():
                    if module is not self.net.last: # -> last layer !
                        for param_name, param in list(module.named_parameters(recurse=False)):
                            delattr(module, param_name)
                            module.register_buffer(param_name, param.detach().data)
                
                self.guide = partial(guide_base,
                    init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(self.net), 
                    init_scale=q_scale)

        else:
            if last_layer > 0:
                raise ValueError("No pretrain file but last_layer True")
            
            self.guide = partial(guide_base, init_scale=q_scale)
        
        self.bnn = VariationalBNN(
            self.net, 
            self.prior, 
            self.likelihood, 
            self.guide, 
            name="bnn"
            )
        self.svi = SVI(
            self.bnn.model,
            self.bnn.guide,
            self.optimizer,
            self.loss
        )
        self.test_preds = {'labels': [], 'preds': [], 'stds': []}



    def test_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1].squeeze()
        with self.fit_ctxt():
            pred = self.bnn.predict(x, num_predictions=100)
        
        acc = (y == torch.argmax(pred, dim=1)).float().mean()
        self.log("acc/test", acc)

        return {"loss": torch.argmax(pred, dim=1), "label": batch[1], 
            "pred": torch.argmax(pred, dim=1)}


    def test_epoch_end(self, outputs):
        for output in outputs:
            self.test_preds['preds'].extend(list(
                output['pred'].flatten().cpu().detach().numpy()))
            self.test_preds['labels'].extend(list(
                output['label'].cpu().detach().numpy()))


    def validation_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1].squeeze()
        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
        self.svi_no_obs = SVI(self.bnn_no_obs, self.bnn.guide,
            self.optimizer, self.loss)


        with self.fit_ctxt():
            elbo = self.svi.evaluate_loss(x, y)
            pred = torch.argmax(self.bnn.predict(x, num_predictions=1), dim = 1)
            #m, sd = pred
            kl = self.svi_no_obs.evaluate_loss(x)
            
        acc = (y == pred).float().mean()
        
        # Permet de calculer la KL sans la likelihood
        
        
        #kl = self.bnn.cached_kl_loss
        self.log('elbo/val', elbo)
        self.log("acc/val", acc)
        self.log('kl/val', kl)
        self.log('likelihood/val', elbo - kl)
        #return {'loss' : mse}


    def training_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1].squeeze()
        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
        self.svi_no_obs = SVI(self.bnn_no_obs, self.bnn.guide,
            self.optimizer, self.loss)
        # As we do not use PyTorch Optimizers, it is needed in order to Pytorch
        # Lightning to know that we are training a model, and initiate routines
        # like checkpointing etc.
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.\
                                    optim_step_progress.increment_ready()
        with self.fit_ctxt():
            elbo = self.svi.step(x, y)
            pred = torch.argmax(self.bnn.predict(x, num_predictions=1), dim = 1)
            kl = self.svi_no_obs.evaluate_loss(x)
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.\
                                    optim_step_progress.increment_completed()
        
        
        acc = (y == pred).float().mean()
        self.log("acc/train", acc)
        self.log("elbo/train", elbo)
        self.log('kl/train', kl)
        self.log('likelihood/train', elbo - kl)
        #return {'loss' : mse}


    def configure_optimizers(self):
        # Because we use Pyro's SVI optimizer that works differently
        #self.optimizer = pyro.optim.Adam({"lr": self.lr, "weight_decay":0,
        #    'betas' : (0.9, 0.999)})
        self.optimizer = pyro.optim.Adagrad({"lr": self.lr})
        return None


    def on_save_checkpoint(self, checkpoint):
        # Saving Pyro's param_store for the bnn's parameters 
        checkpoint["param_store"] = pyro.get_param_store().get_state()
        

    def on_load_checkpoint(self, checkpoint):
        pyro.get_param_store().set_state(checkpoint["param_store"])
        checkpoint['state_dict'] = remove_dict_entry_startswith(
            checkpoint['state_dict'], 'bnn')


    @staticmethod
    def add_model_specific_args(parent_parser):
        # To initialize from checkpoint, without giving init args 
        parser = parent_parser.add_argument_group("NCMAPSSModelBnn")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser
"""