from abc import ABC, abstractmethod

from pytorch_lightning import LightningDataModule



class Inference(ABC):
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
    @abstractmethod
    def num_params(self) -> int:
        ...