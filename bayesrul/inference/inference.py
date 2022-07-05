from abc import ABC, abstractmethod

from pytorch_lightning import LightningDataModule



class Inference(ABC):
    """ Abstract class used to simplify benchmarking.
        Provided a LightningDataModule, initializes a model and offers methods
        to train, test and compute uncertainties on test set.
            data = LightningDataModule(...)
            inference = ...
            inference.fit(2)
            inference.test()
    """
    name: str = None

    @abstractmethod
    def __init__(
        self,
        args,
        data: LightningDataModule,
        hyperparams = None,
        GPU = 1,
    ) -> None:
        ...

    @abstractmethod
    def _define_model(
        self
    ):
        ...

    @abstractmethod
    def fit(
        self,
        epochs: int,
    ):
        ...

    @abstractmethod
    def test(
        self,
    ):
        ...

    @abstractmethod
    def epistemic_aleatoric_uncertainty(self):
        ...

    @property
    def _args(self):
        return(self.args)

    @property
    @abstractmethod
    def num_params(self) -> int:
        ...