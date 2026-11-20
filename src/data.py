"""Load CIFAKE (real CIFAR-10 photos vs Stable Diffusion 1.4 images, 32x32).

The parquet files are downloaded from the Hugging Face mirror the first time and cached as
uint8 arrays in data/cifake.npz. Label 1 = AI-generated (fake), 0 = real.
"""
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = "https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset/resolve/main/data/"
FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}


def _parquet(split):
    p = DATA / f"{split}.parquet"
    if not p.exists():
        DATA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(BASE + FILES[split], p)
    return pd.read_parquet(p)


def _to_arrays(df):
    x = np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["image"]])
    # the mirror uses 0 = FAKE, 1 = REAL; flip so 1 means "AI-generated"
    y = (df["label"].to_numpy() == 0).astype(np.int64)
    return x, y


def load():
    """Return (x_train, y_train, x_test, y_test); x is uint8 (N, 32, 32, 3)."""
    cache = DATA / "cifake.npz"
    if cache.exists():
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    xtr, ytr = _to_arrays(_parquet("train"))
    xte, yte = _to_arrays(_parquet("test"))
    np.savez_compressed(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte


def split_val(x, y, frac=0.1, seed=0):
    idx = np.random.default_rng(seed).permutation(len(x))
    n = int(len(x) * frac)
    return x[idx[n:]], y[idx[n:]], x[idx[:n]], y[idx[:n]]
