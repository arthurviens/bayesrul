from pathlib import Path

import torch

import pytorch_lightning as pl

from bayesrul.inference.inference import Inference
from bayesrul.lightning_wrappers.frequentist import DnnPretrainWrapper
from bayesrul.utils.miscellaneous import get_checkpoint, TBLogger
from bayesrul.utils.plotting import PredLogger
from bayesrul.lightning_wrappers.bayesian import VIBnnWrapper


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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
    ) -> None:
        self.name = f"vi_{args.model_name}_{args.archi}_{args.guide}"
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
                'last_layer': args.last_layer,
                'pretrain_file' : None,
                'device' : torch.device(f"cuda:{self.GPU}"),
            }
            
        if hyperparams is not None: # Overriding defaults with arguments
            for key in hyperparams.keys():
                hyp[key] = hyperparams[key]

        self.args = Dotdict({**(args.__dict__), **hyp}) # Merge dicts

        if self.args.guide == "radial": hyperparams['fit_context'] = 'null'

        self.base_log_dir = Path(args.out_path, "bayesian", args.model_name)
        self.checkpoint_file = get_checkpoint(self.base_log_dir, version=None)
    
        self.logger = TBLogger(
            Path(self.base_log_dir),
            default_hp_metric=False,
        )

    def fit(self, epochs):

        self.trainer = pl.Trainer(
            default_root_dir=self.base_log_dir,
            gpus=[self.GPU], 
            max_epochs=epochs,
            log_every_n_steps=2,
            logger=self.logger,
        )

        checkpoint_file = get_checkpoint(self.base_log_dir, version=None)
        if self.args['pretrain'] > 0 and checkpoint_file:
            raise ValueError("Can not pretrain and resume from checkpoint")

        if self.args.pretrain > 0 and (not checkpoint_file):
            pre_net = DnnPretrainWrapper(self.data.win_length, self.data.n_features,
                archi = self.args.archi, bias=self.hyperparams['bias'])
            pre_trainer = pl.Trainer(gpus=[self.GPU], max_epochs=self.args.pretrain, logger=False,
                checkpoint_callback=False)

            pretrain_dir = Path(self.base_log_dir, "lightning_logs",
                f'version_{self.trainer.logger.version}')
            pretrain_dir.mkdir(exist_ok=True, parents=True)
            self.hyperparams['pretrain_file'] = Path(pretrain_dir,
                f'pretrained_{self.args.pretrain}.pt').as_posix()
            pre_trainer.fit(pre_net, self.data)
            self.base_log_dir.mkdir(exist_ok=True)
            torch.save(pre_net.net.state_dict(), self.hyperparams['pretrain_file'])
            
        if checkpoint_file:
            self.dnn = VIBnnWrapper.load_from_checkpoint(checkpoint_file,
                map_location=self.args.device)
        else:
            self.dnn = VIBnnWrapper(
                self.data.win_length, 
                self.data.n_features, 
                self.data.train_size,
                **self.args
            )

        self.trainer.fit(self.dnn, self.data)

    def test(self):

        tester = pl.Trainer(
            gpus=[self.GPU], 
            log_every_n_steps=10, 
            logger=self.logger, 
            max_epochs=-1 # Silence warning
            ) 
        tester.test(self.dnn, self.data, verbose=False)

        predLog = PredLogger(self.base_log_dir)
        predLog.save(self.dnn.test_preds)

    def epistemic_aleatoric_uncertainty(self):
        self.dnn.test_preds
        ...

    def num_params(self) -> int:
        ...