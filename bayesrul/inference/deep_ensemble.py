from pathlib import Path

import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from bayesrul.inference.inference import Inference
from bayesrul.lightning_wrappers.frequentist import DeepEnsembleWrapper
from bayesrul.utils.miscellaneous import (
    get_checkpoint,
    TBLogger,
    Dotdict,
    numel,
)
from bayesrul.utils.post_process import ResultSaver


class DeepEnsemble(Inference):
    """
    Deep Ensemble neural networks inference class
    """

    def __init__(
        self,
        args,
        data: pl.LightningDataModule,
        n_models: int,
        hyperparams=None,
        GPU=0,
        studying=False,
    ) -> None:

        assert isinstance(GPU, int), f"GPU argument should be an int, not {type(GPU)}"
        assert isinstance(
            n_models, int
        ), f"n_models argument should be an int, not {type(n_models)}"
        assert isinstance(
            data, pl.LightningDataModule
        ), f"data argument should be a LightningDataModule, not {type(data)}"
        self.data = data
        self.GPU = GPU
        self.n_models = n_models

        hyp = {
            "bias": True,
            "lr": 1e-3,
            "device": torch.device(f"cuda:{self.GPU}"),
            "dropout": 0,
            "out_size": 2,
        }

        if hyperparams is not None:  # Overriding defaults with arguments
            for key in hyperparams.keys():
                hyp[key] = hyperparams[key]

        # Merge dicts and make attributes accessible by .
        self.args = Dotdict({**(args.__dict__), **hyp})
        try:
            del self.args["n_models"]
        except KeyError:
            pass

        directory = "studies" if studying else "frequentist"
        self.base_log_dir = Path(args.out_path, directory, args.model_name)

        self.checkpoint_file = get_checkpoint(self.base_log_dir, version=None)

        self.logger = TBLogger(
            Path(self.base_log_dir),
            default_hp_metric=False,
        )

    def _define_model(self):
        self.checkpoint_file = get_checkpoint(self.base_log_dir, version=None)
        if self.checkpoint_file:
            print("Loading model from checkpoint")
            self.dnn = DeepEnsembleWrapper.load_from_checkpoint(
                self.checkpoint_file, map_location=self.args.device
            )
        else:
            print(self.args)
            self.dnn = DeepEnsembleWrapper(
                self.data.win_length,
                self.data.n_features,
                self.n_models,
                **self.args,
            )

    def fit(self, epochs, monitors=None, early_stop=0):
        if not hasattr(self, "dnn"):
            self._define_model()

        self.trainer = pl.Trainer(
            default_root_dir=self.base_log_dir,
            gpus=[self.GPU],
            max_epochs=epochs,
            log_every_n_steps=2,
            logger=self.logger,
            callbacks=[
                EarlyStopping(monitor="gaussian_nll/val", patience=early_stop),
            ]
            if early_stop
            else None,
        )

        self.trainer.fit(self.dnn, self.data, ckpt_path=self.checkpoint_file)

        return self.trainer.callback_metrics["mse/val"]

    def test(self):
        if not hasattr(self, "dnn"):
            self._define_model()

        tester = pl.Trainer(
            gpus=[self.GPU], log_every_n_steps=10, logger=self.logger, max_epochs=-1
        )  # Silence warning

        tester.test(self.dnn, self.data, verbose=False)

        self.results = ResultSaver(self.base_log_dir)
        self.results.save(self.dnn.test_preds)

    def epistemic_aleatoric_uncertainty(self, device=None):
        raise NotImplementedError("Deep Ensembles can't model epistemic uncertainties.")

    def num_params(self) -> int:
        if not hasattr(self, "dnn"):
            self._define_model()

        return np.sum([numel(x) for x in self.dnn.nets])
