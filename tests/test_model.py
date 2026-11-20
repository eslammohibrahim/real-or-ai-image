import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import split_val  # noqa: E402
from model import SmallCNN, predict_proba, to_tensor  # noqa: E402


def test_to_tensor_shape():
    x = np.zeros((4, 32, 32, 3), np.uint8)
    assert to_tensor(x).shape == (4, 3, 32, 32)


def test_model_outputs_probabilities():
    torch.manual_seed(0)
    p = predict_proba(SmallCNN(), np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8))
    assert p.shape == (5,) and ((p >= 0) & (p <= 1)).all()


def test_split_val_no_overlap():
    x = np.arange(100)[:, None]
    y = np.zeros(100)
    xa, _, xb, _ = split_val(x, y, frac=0.2)
    assert len(xb) == 20 and not set(xa[:, 0]) & set(xb[:, 0])
