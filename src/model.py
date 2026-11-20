import numpy as np
import torch
from torch import nn

MEAN = torch.tensor([0.48, 0.47, 0.43]).view(1, 3, 1, 1)
STD = torch.tensor([0.24, 0.23, 0.25]).view(1, 3, 1, 1)


def to_tensor(x_uint8):
    """(N, 32, 32, 3) uint8 -> normalized float tensor (N, 3, 32, 32)."""
    t = torch.from_numpy(np.ascontiguousarray(x_uint8)).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN) / STD


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Three conv blocks (32 -> 16 -> 8 -> 4) and a linear head. About 0.2M parameters."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 96))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def predict_proba(model, x_uint8, batch=1000):
    """Probability that each image is AI-generated."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_uint8), batch):
            out.append(torch.sigmoid(model(to_tensor(x_uint8[i:i + batch]))).numpy())
    return np.concatenate(out)
