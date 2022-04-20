import pickle
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, TypeVar, Union

import lmdb
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from typing_extensions import Protocol

T = TypeVar("T", contravariant=True)


class Aggregate(Protocol[T]):
    def feed(self, line: T, i: int) -> None:
        ...

    def get(self) -> Dict[str, Union[np.ndarray, bytearray]]:
        ...


def create_lmdb(
    filename: Path,
    iterator: Iterator[T],
    patterns: Dict[str, Callable[[T], Union[np.ndarray, bytearray]]],
    aggs: List[Aggregate[T]] = [],
    map_size=1024**4,
    **kwargs,  # silence
) -> None:

    env = lmdb.open(  # type: ignore
        filename.as_posix(), readonly=False, map_size=map_size
    )

    with env.begin(write=True) as txn:

        for i, line in enumerate(iterator):

            for agg in aggs:
                agg.feed(line, i)

            for key, value in patterns.items():
                txn.put(key.format(i).encode(), value(line))

            txn.put(b"nb_lines", str(i + 1).encode())

        for agg in aggs:
            for key, cumul in agg.get().items():
                txn.put(key.encode(), cumul)

        for key, arg_value in kwargs.items():
            txn.put(key.encode(), str(arg_value).encode())


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class MinMaxScalerTransform:
    def __init__(self, dataset):
        self.min = dataset.get("min_sample")
        max_ = dataset.get("max_sample")
        self.range = (max_ - self.min) + 1e-12

    def __call__(self, x):
        return (x - self.min) / self.range


class PCATransform:
    def __init__(self, dataloader, n_features, n_components, seq_padding, data_path):
        l = [t for t, l in dataloader] if seq_padding else [t for t in dataloader]
        batch_data = (
            torch.cat(l[:-1], dim=1).squeeze().numpy()
        )  # dropping the last batch
        all_data = batch_data.reshape(-1, n_features)
        # all_data = torch.cat([t for t in dataloader], dim=1).squeeze().numpy()
        pca_path = Path(f"{data_path}/pca.pkl")
        if pca_path.exists():
            with open(pca_path, "rb") as pca_file:
                self.pca = pickle.load(pca_file)
        else:
            self.pca = PCA(n_components=n_components, svd_solver="full")
            self.pca.fit(all_data)
            with open(pca_path, "wb") as pca_file:
                pickle.dump(self.pca, pca_file)
        self.n_features = n_features
        self.n_components = n_components

    def inverse_transform(self, x):
        return (
            torch.from_numpy(
                self.pca.inverse_transform(x.cpu().reshape(-1, self.n_components))
            )
            .reshape(x.shape[0], -1, self.n_features)
            .to(x.get_device())
        )

    def __call__(self, x):
        return self.pca.transform(x)


class LmdbDataset(Dataset):
    def __init__(
        self,
        path: Union[Path, str],
        pattern: str,
        transform: Optional[Callable] = None,
    ):
        self.env = lmdb.open(Path(path).as_posix(), readonly=True)
        self.txn = self.env.begin(write=False)
        self.len = int(self.get("nb_lines", numpy=False))
        self.pattern = pattern
        self.dtype = (
            np.float32 if int(self.get("bits", numpy=False)) == 32 else np.float64
        )
        self.n_features = int(self.get("n_features", numpy=False))
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        buffer = self.txn.get(self.pattern.format(i).encode())
        sample = np.frombuffer(buffer, dtype=self.dtype).reshape(self.n_features, -1).T
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get(self, key: str, numpy=True, dtype=None):
        value = self.txn.get(key.encode())
        if numpy:
            if dtype is None:
                return np.frombuffer(value, dtype=self.dtype)
            else:
                return np.frombuffer(value, dtype=dtype)
        else:
            return value.decode() if value is not None else None
