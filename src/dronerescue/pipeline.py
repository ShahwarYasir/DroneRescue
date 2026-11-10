"""Load the cleaned file list and the cached image arrays."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def load_manifest_clean(data_dir=None):
    data_dir = Path(data_dir or C.DATA_DIR)
    path = data_dir / "manifest_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/01_eda_dedupe.py first.")
    return pd.read_csv(path)


def cache_dir(data_dir, size, variant="cleaned"):
    """'cleaned' = the de-duplicated split used for all main results; 'original' = the official split as published."""
    return Path(data_dir) / (f"cache_{size}" if variant == "cleaned" else f"cache_{size}_{variant}")


def load_split(data_dir, size, split, variant="cleaned"):
    """Return x (N,size,size,3 uint8), y (N,) int labels, exposed (N,) bool and the file paths."""
    f = cache_dir(data_dir, size, variant) / f"{split}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run scripts/03_build_cache.py --size {size} --variant {variant} first.")
    z = np.load(f)
    return z["x"], z["y"], z["exposed"], z["paths"]


def class_weights(y):
    """'Balanced' class weights: total / (n_classes * count). Rare classes get weights above 1."""
    counts = np.bincount(y, minlength=len(C.CLASSES)).astype(float)
    w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return {i: float(w[i]) for i in range(len(counts))}
