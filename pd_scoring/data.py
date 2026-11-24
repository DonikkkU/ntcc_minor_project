
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import FEATURES_DIR, N_PARTS

TRAIN_PARTS = list(range(0, 8))   
VALID_PARTS = [8, 9]              
OOT_PARTS = [10, 11]              

def part_path(i: int):
    return FEATURES_DIR / f"part_{i:02d}.parquet"


def all_feature_columns() -> list[str]:
    cols = pq.ParquetFile(part_path(0)).schema_arrow.names
    return [c for c in cols if c not in ("id", "flag")]


def load(parts, columns, dtype=np.float32, sample_frac: float | None = None, seed: int = 0):
    """Return (X, y, ids) for the given partitions.

    X is a preallocated ``dtype`` matrix filled partition by partition, so the
    peak memory is roughly the size of X plus one partition.
    """
    rng = np.random.default_rng(seed)
    pieces_idx, sizes = [], []
    for p in parts:
        n = pq.ParquetFile(part_path(p)).metadata.num_rows
        idx = np.arange(n) if sample_frac is None else np.sort(rng.choice(n, int(n * sample_frac), replace=False))
        pieces_idx.append(idx)
        sizes.append(len(idx))
    X = np.empty((sum(sizes), len(columns)), dtype=dtype)
    y = np.empty(sum(sizes), dtype=np.uint8)
    ids = np.empty(sum(sizes), dtype=np.int64)
    pos = 0
    for p, idx in zip(parts, pieces_idx):
        t = pq.read_table(part_path(p), columns=["id", "flag"] + list(columns))
        df = t.to_pandas()
        del t
        if sample_frac is not None:
            df = df.iloc[idx]
        k = len(df)
        X[pos:pos + k] = df[list(columns)].to_numpy(dtype=dtype)
        y[pos:pos + k] = df["flag"].to_numpy()
        ids[pos:pos + k] = df["id"].to_numpy()
        pos += k
        del df
    return X, y, ids


def load_frame(parts, columns) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(part_path(p), columns=["id", "flag"] + list(columns)) for p in parts],
                     ignore_index=True)


try:
    import lightgbm as lgb
    class ArraySequence(lgb.Sequence):

        def __init__(self, arr: np.ndarray, batch_size: int = 32768):
            self.arr = arr
            self.batch_size = batch_size

        def __getitem__(self, idx):
            return np.asarray(self.arr[idx], dtype=np.float64)

        def __len__(self):
            return len(self.arr)
except ImportError: 
    pass
