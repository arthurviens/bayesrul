import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..utils.lmdb_utils import LmdbDataset


class CMAPSSLmdbDataset(LmdbDataset):
    def __getitem__(self, i: int):
        sample = super().__getitem__(i)
        rul = self.dtype(super().get(f"rul_{i}", numpy=False))
        return sample.copy(), rul


class CMAPSSLmdbDatasetAll(CMAPSSLmdbDataset):
    def __getitem__(self, i: int):
        sample, rul = super().__getitem__(i)
        ds_id = int(super().get(f"ds_id_{i}", numpy=False))
        traj_id = int(super().get(f"traj_id_{i}", numpy=False))
        win_id = int(super().get(f"win_id_{i}", numpy=False))
        settings = super().get(f"settings_{i}", numpy=True)
        return ds_id, traj_id, win_id, settings.copy(), rul, sample


class CMAPSSDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.datasets = dict(
            (
                name,
                CMAPSSLmdbDataset(
                    f"{self.data_path}/lmdb/{name}.lmdb",
                    "{}",
                ),
            )
            for name in ["train", "val", "test"]
        )
        self.win_length, self.n_features = (
            int(self.datasets["train"].get("win_length", numpy=False)),
            self.datasets["train"].n_features,
        )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=10,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=10,
            pin_memory=True,
        )
