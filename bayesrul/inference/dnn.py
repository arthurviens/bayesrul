
from pathlib import Path

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from bayesrul.inference.inference import Inference
from bayesrul.ncmapss.frequentist import NCMAPSSModel
from bayesrul.ncmapss.frequentist import get_checkpoint, TBLogger
from bayesrul.utils.plotting import PredLogger


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DNN(Inference):
    """
    Standard frequentist neural networks
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
            'lr' : 1e-3,
            'device' : torch.device(f"cuda:{self.GPU}"),
        }
            
        if hyperparams is not None: # Overriding defaults with arguments
            for key in hyperparams.keys():
                hyp[key] = hyperparams[key]

        self.args = Dotdict({**(args.__dict__), **hyp}) # Merge dicts

        self.base_log_dir = Path(args.out_path, "frequentist", args.model_name)

        self.checkpoint_file = get_checkpoint(self.base_log_dir, version=None)

        self.logger = TBLogger(
            Path(self.base_log_dir),
            default_hp_metric=False,
        )


    def fit(self, epochs):
        dnn = NCMAPSSModel(
            self.data.win_length, 
            self.data.n_features, 
            **self.args,
        )

        self.monitor = f"{dnn.loss}/val"
        earlystopping_callback = EarlyStopping(monitor=self.monitor, patience=50)

        self.trainer = pl.Trainer(
            default_root_dir=self.base_log_dir,
            gpus=[1],
            max_epochs=epochs,
            log_every_n_steps=2,
            logger=self.logger,
            callbacks=[
                earlystopping_callback,
            ],
        )


        self.trainer.fit(dnn, self.data, ckpt_path=self.checkpoint_file)



    def test(self):
        checkpoint_file = get_checkpoint(self.base_log_dir, version=None)

        dnn = NCMAPSSModel.load_from_checkpoint(checkpoint_file)
        tester = pl.Trainer(
            gpus=[0], 
            log_every_n_steps=10, 
            logger=self.logger, 
            max_epochs=-1
        ) # Silence warning
        
        tester.test(dnn, self.data, verbose=False)
        predLog = PredLogger(self.base_log_dir)
        predLog.save(dnn.test_preds)


    def epistemic_aleatoric_uncertainty(self):
        self.dnn.test_preds
        ...

    def num_params(self) -> int:
        ...