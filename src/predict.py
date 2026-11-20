"""python src/predict.py path/to/image.jpg [more images...]

The model was trained on 32x32 images, so every image is shrunk to 32x32 first.
That means it is only meaningful for images like the ones it was trained on.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data import ROOT  # noqa: E402
from model import SmallCNN, predict_proba  # noqa: E402


def load_model():
    m = SmallCNN()
    m.load_state_dict(torch.load(ROOT / "models/cnn.pt"))
    return m


def prepare(path):
    return np.array(Image.open(path).convert("RGB").resize((32, 32), Image.BICUBIC))


if __name__ == "__main__":
    model = load_model()
    for f in sys.argv[1:]:
        p = float(predict_proba(model, prepare(f)[None])[0])
        print(f"{f}: {'AI-generated' if p > 0.5 else 'real'} (p(AI) = {p:.2f})")
