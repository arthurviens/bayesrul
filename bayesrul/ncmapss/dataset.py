import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..utils.lmdb_utils import LmdbDataset

class NCMAPSSLmdbDataset(LmdbDataset):
    def __getitem__(self, i: int):
        sample = super().__getitem__(i)
        rul = self.dtype(super().get(f"rul_{i}", numpy=False))
        return sample.copy(), rul


class NCMAPSSLmdbDatasetAll(NCMAPSSLmdbDataset):
    def __getitem__(self, i: int):
        sample, rul = super().__getitem__(i)
        ds_id = int(super().get(f"ds_id_{i}", numpy=False))
        traj_id = int(float(super().get(f"unit_id_{i}", numpy=False)))
        win_id = int(super().get(f"win_id_{i}", numpy=False))
        settings = super().get(f"settings_{i}", numpy=True)
        return ds_id, traj_id, win_id, settings.copy(), rul, sample


class NCMAPSSDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, all_dset=False):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        if not all_dset:
            self.datasets = dict(
                (
                    name,
                    NCMAPSSLmdbDataset(
                        f"{self.data_path}/lmdb/{name}.lmdb",
                        "{}",
                    ),
                )
                for name in ["train", "val", "test"]
            )
        else:
            self.datasets = dict(
                (
                    name,
                    NCMAPSSLmdbDatasetAll(
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

    @property
    def train_size(self):
        return len(self.datasets['train'])

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=False, # Important
            num_workers=2,
            pin_memory=True,
        )
