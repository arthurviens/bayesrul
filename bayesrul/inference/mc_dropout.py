from pathlib import Path

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from bayesrul.inference.inference import Inference
from bayesrul.lightning_wrappers.frequentist import DnnWrapper
from bayesrul.utils.miscellaneous import (
    get_checkpoint,
    TBLogger,
    Dotdict,
    enable_dropout,
    numel,
)
from bayesrul.utils.post_process import ResultSaver
from tqdm import tqdm


def assert_dropout(model):
    """Verifies that the model contains dropout layers"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            return True
    return False


class MCDropout(Inference):
    """
    MC Dropout neural network inference class
    """

    def __init__(
        self,
        args,
        data: pl.LightningDataModule,
        p_dropout: float,
        hyperparams=None,
        GPU=0,
        studying=False,
    ) -> None:
        assert isinstance(GPU, int), f"GPU argument should be an int, not {type(GPU)}"
        assert isinstance(
            data, pl.LightningDataModule
        ), f"data argument should be a LightningDataModule, not {type(data)}"
        self.data = data
        self.GPU = GPU

        hyp = {
            "bias": True,
            "lr": 1e-3,
            "device": torch.device(f"cuda:{self.GPU}"),
            "dropout": p_dropout,
            "out_size": 2,
        }

        if hyperparams is not None:  # Overriding defaults with arguments
            for key in hyperparams.keys():
                hyp[key] = hyperparams[key]

        # Merge dicts and make attributes accessible by .
        self.args = Dotdict({**(args.__dict__), **hyp})

        directory = "studies" if studying else "frequentist"
        self.base_log_dir = Path(args.out_path, directory, args.model_name)

        self.checkpoint_file = get_checkpoint(self.base_log_dir, version=None)

        self.logger = TBLogger(
            Path(self.base_log_dir),
            default_hp_metric=False,
        )

    def _define_model(self, device=None):
        if device is not None:
            self.args.device = device
        self.checkpoint_file = get_checkpoint(self.base_log_dir, version=None)
        if self.checkpoint_file:
            print(f"Loading model from checkpoint into {self.args.device}")
            self.dnn = DnnWrapper.load_from_checkpoint(self.checkpoint_file)
        else:
            self.dnn = DnnWrapper(
                self.data.win_length,
                self.data.n_features,
                **self.args,
            )
        self.dnn.to_device(self.args.device)

        assert assert_dropout(self.dnn), "MCDropout Model has no dropout layers"

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
                EarlyStopping(monitor=f"{self.dnn.loss}/val", patience=early_stop),
            ]
            if early_stop
            else None,
        )

        self.trainer.fit(self.dnn, self.data, ckpt_path=self.checkpoint_file)

        return self.trainer.callback_metrics["mse/val"]

    def test(self, device=None):
        if not hasattr(self, "dnn"):
            self._define_model(device=device)

        enable_dropout(self.dnn)

        tester = pl.Trainer(
            gpus=[self.GPU], log_every_n_steps=10, logger=self.logger, max_epochs=-1
        )  # Silence warning

        tester.test(self.dnn, self.data, verbose=False)

        self.results = ResultSaver(self.base_log_dir)
        self.results.save(self.dnn.test_preds)

    def epistemic_aleatoric_uncertainty(self, device=None):
        if device is None:
            device = self.args.device
        if not hasattr(self, "dnn"):
            print("Redefining model ?")
            self._define_model(device=device)

        self.dnn.to_device(device)

        dim = 0
        n = 0
        pred_loc, predictive_var, epistemic_var, aleatoric_var = [], [], [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(self.data.test_dataloader())):
                x, y = x.to(device), y.to(device)
                n += len(x)

                outputs = []
                for j in range(10):
                    outputs.append(self.dnn.forward(x))
                outputs = torch.stack(outputs)

                if isinstance(outputs, tuple):
                    loc, scale = outputs
                else:
                    outputs = outputs.squeeze()
                    loc, scale = outputs[:, :, 0], outputs[:, :, 1]

                outputs = []

                # Sd is the PREDICTIVE VARIANCE
                pred_loc.append(loc.mean(axis=dim))

                predictive_var.append((scale**2).mean(dim).add(loc.var(dim)).cpu())
                # predictive_unc = predictive_var.sqrt()
                epistemic_var.append(loc.var(dim).cpu())
                # epistemic_unc = epistemic_var.sqrt()
                aleatoric_var.append((scale**2).mean(dim).cpu())
                # aleatoric_unc = aleatoric_var.sqrt()

        pred_loc = torch.cat(pred_loc)
        predictive_var = torch.cat(predictive_var)
        epistemic_var = torch.cat(epistemic_var)
        aleatoric_var = torch.cat(aleatoric_var)

        assert (
            len(pred_loc) == n
        ), f"Size ({len(pred_loc)}) of uncertainties should match {n}"
        assert (
            len(predictive_var) == n
        ), f"Size ({len(predictive_var)}) of uncertainties should match {n}"
        assert (
            len(epistemic_var) == n
        ), f"Size ({len(epistemic_var)}) of uncertainties should match {n}"
        assert (
            len(aleatoric_var) == n
        ), f"Size ({len(aleatoric_var)}) of uncertainties should match {n}"

        self.results.append(
            {  # Automatic save to file
                "unweighted_pred_loc": pred_loc.cpu().numpy(),
                "pred_var": predictive_var.cpu().numpy(),
                "ep_var": epistemic_var.cpu().numpy(),
                "al_var": aleatoric_var.cpu().numpy(),
            }
        )

    def num_params(self) -> int:
        if not hasattr(self, "dnn"):
            self._define_model()

        return numel(self.dnn.net)
