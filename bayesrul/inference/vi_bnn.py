from pathlib import Path
from tqdm import tqdm

import torch

import pytorch_lightning as pl

from bayesrul.inference.inference import Inference
from bayesrul.lightning_wrappers.frequentist import DnnPretrainWrapper
from bayesrul.lightning_wrappers.bayesian import VIBnnWrapper
from bayesrul.utils.miscellaneous import get_checkpoint, TBLogger, Dotdict
from bayesrul.utils.plotting import ResultSaver

import pyro 

DEBUG=True

class VI_BNN(Inference):
    """
    Bayesian Neural Network with Variational Inference
    """

    def __init__(
        self,
        args,
        data: pl.LightningDataModule,
        hyperparams = None,
        GPU = 1,
        studying = False,
    ) -> None:
        assert isinstance(GPU, int), \
            f"GPU argument should be an int, not {type(GPU)}"
        assert isinstance(data, pl.LightningDataModule), \
            f"data argument should be a LightningDataModule, not {type(data)}"
        self.data = data
        self.GPU = GPU

        hyp = {
                'bias' : True,
                'prior_loc' : 0,
                'prior_scale' : 0.5,
                'likelihood_scale' : 0.001, # Useless in Heteroskedastic case
                'q_scale' : 0.01,
                'fit_context' : 'lrt',
                'num_particles' : 1,
                'optimizer': 'nadam',
                'lr' : 1e-3,
                'out_size': 2,
                'last_layer': False,
                'pretrain': 0,
                'pretrain_file': None,
                'device' : torch.device(f"cuda:{self.GPU}"),
            }
            
        if hyperparams is not None: # Overriding defaults with arguments
            for key in hyperparams.keys():
                hyp[key] = hyperparams[key]

        # Merge dicts and make attributes accessible by .
        self.args = Dotdict({**(args.__dict__), **hyp}) 
        if DEBUG: print(f"Training parameters : {self.args}")

        if self.args.guide == "radial": hyperparams['fit_context'] = 'null'

        directory = "studies" if studying else "bayesian"
        self.base_log_dir = Path(args.out_path, directory, args.model_name)
        self.checkpoint_file = get_checkpoint(self.base_log_dir, version=None)
    
        self.logger = TBLogger(
            Path(self.base_log_dir),
            default_hp_metric=False,
        )

    def _define_model(self):
        checkpoint_file = get_checkpoint(self.base_log_dir, version=None)
        if self.args['pretrain'] > 0 and checkpoint_file:
            raise ValueError("Can not pretrain and resume from checkpoint")

        if self.args.pretrain > 0 and (not checkpoint_file):
            pre_net = DnnPretrainWrapper(self.data.win_length, self.data.n_features,
                archi = self.args.archi, bias=self.args['bias'])
            pre_trainer = pl.Trainer(gpus=[self.GPU], max_epochs=self.args.pretrain, logger=False,
                checkpoint_callback=False)

            pretrain_dir = Path(self.base_log_dir, "lightning_logs",
                f'version_{self.trainer.logger.version}')
            pretrain_dir.mkdir(exist_ok=True, parents=True)
            self.args['pretrain_file'] = Path(pretrain_dir,
                f'pretrained_{self.args.pretrain}.pt').as_posix()
            pre_trainer.fit(pre_net, self.data)
            self.base_log_dir.mkdir(exist_ok=True)
            torch.save(pre_net.net.state_dict(), self.args['pretrain_file'])

            
        if checkpoint_file:
            self.args.device = torch.device(f'cuda:{self.GPU}')
            self.bnn = VIBnnWrapper.load_from_checkpoint(checkpoint_file,
                map_location=self.args.device)
        else:
            self.bnn = VIBnnWrapper(
                self.data.win_length, 
                self.data.n_features, 
                self.data.train_size,
                **self.args
            )

    def fit(self, epochs: int, monitors=None):
        if ((monitors is not None) & 
            (('bi_obj' not in self.base_log_dir.as_posix()) 
                or ('single_obj' not in self.base_log_dir.as_posix()))):
            base = "/".join(self.base_log_dir.as_posix().split('/')[:-1])
            end = self.base_log_dir.as_posix().split('/')[-1]
            if len(monitors) == 1:
                log_dir = Path(base, "single_obj", end)
            else:
                log_dir = Path(base, "bi_obj", end)
            self.base_log_dir = log_dir
        self.trainer = pl.Trainer(
            default_root_dir=self.base_log_dir,
            gpus=[self.GPU], 
            max_epochs=epochs,
            log_every_n_steps=2,
            logger=self.logger,
        )

        if not hasattr(self, 'bnn'):
            self._define_model()
        
        if epochs > 0:
            self.trainer.fit(self.bnn, self.data)
            if monitors:
                return [self.trainer.callback_metrics[monitor] for monitor in monitors]
        else:
            raise RuntimeError(f"Cannot fit model for {epochs} epochs.")


    def test(self):
        tester = pl.Trainer(
            gpus=[self.GPU], 
            log_every_n_steps=10, 
            logger=self.logger, 
            max_epochs=-1 # Silence warning
        ) 
        
        tester.test(self.bnn, self.data, verbose=False)

        self.results = ResultSaver(self.base_log_dir)
        self.results.save(self.bnn.test_preds)

    def epistemic_aleatoric_uncertainty(self, device=None):
        if device is None:
            device = self.args.device

        dim = 0

        n = 0
        pred_loc, predictive_var, epistemic_var, aleatoric_var = [], [], [], []
        for i, (x, y) in enumerate(tqdm(self.data.test_dataloader())):
            x, y = x.to(device), y.to(device)
            n += len(x)

            output = self.bnn.bnn.predict(x, num_predictions=10, aggregate=False)
            if isinstance(output, tuple):
                loc, scale = output
            else:
                output = output.squeeze()
                loc, scale = output[:, :, 0], output[:, :, 1]
            # Sd is the PREDICTIVE VARIANCE 
            pred_loc.append(loc.mean(axis=dim))

            predictive_var.append((scale**2).mean(dim).add(loc.var(dim)))
            #predictive_unc = predictive_var.sqrt()
            epistemic_var.append(loc.var(dim))
            #epistemic_unc = epistemic_var.sqrt()
            aleatoric_var.append((scale**2).mean(dim))
            #aleatoric_unc = aleatoric_var.sqrt()

        pred_loc = torch.cat(pred_loc)
        predictive_var = torch.cat(predictive_var)
        epistemic_var = torch.cat(epistemic_var)
        aleatoric_var = torch.cat(aleatoric_var)

        assert len(pred_loc) == n, f"Size ({len(pred_loc)}) of uncertainties should match {n}"
        assert len(predictive_var) == n, f"Size ({len(predictive_var)}) of uncertainties should match {n}"
        assert len(epistemic_var) == n, f"Size ({len(epistemic_var)}) of uncertainties should match {n}"
        assert len(aleatoric_var) == n, f"Size ({len(aleatoric_var)}) of uncertainties should match {n}"

        self.results.append({ # Automatic save to file
            'unweighted_pred_loc': pred_loc.detach().cpu().numpy(),
            'pred_var': predictive_var.detach().cpu().numpy(),
            'ep_var': epistemic_var.detach().cpu().numpy(),
            'al_var': aleatoric_var.detach().cpu().numpy(),
        })
        
        
    def remove_pretrain_file(self):
        checkpoint_file = get_checkpoint(self.base_log_dir, version=None)
        print(f"Checkpoint_file : {checkpoint_file}")


    def num_params(self) -> int:
        ...