"""Train the baseline (logistic regression on pixels) and the small CNN.

python src/train.py            -> models/cnn.pt, reports/train_log.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from data import ROOT, load, split_val  # noqa: E402
from model import SmallCNN, predict_proba, to_tensor  # noqa: E402

EPOCHS = 8
torch.manual_seed(0)
torch.set_num_threads(8)


def baseline(xtr, ytr, xva, yva, n=20000):
    """Logistic regression on raw pixels, trained on a 20k subset. Shows how far 'just colors' gets you."""
    f = lambda x: x.reshape(len(x), -1).astype(np.float32) / 255.0  # noqa: E731
    clf = LogisticRegression(max_iter=300, C=0.1)
    clf.fit(f(xtr[:n]), ytr[:n])
    return float((clf.predict(f(xva)) == yva).mean())


def augment(xb):
    flip = torch.rand(len(xb)) < 0.5
    xb[flip] = xb[flip].flip(3)
    return xb


def main():
    xtr, ytr, _, _ = load()
    xtr, ytr, xva, yva = split_val(xtr, ytr)
    log = {"train": len(xtr), "val": len(xva), "baseline_val_acc": baseline(xtr, ytr, xva, yva)}
    print("baseline", log["baseline_val_acc"])

    model = SmallCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    steps = EPOCHS * (len(xtr) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    loss_fn = nn.BCEWithLogitsLoss()
    yt = torch.from_numpy(ytr).float()
    best, hist, t0 = 0.0, [], time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr) - 127, 128):
            idx = perm[i:i + 128].numpy()
            xb = augment(to_tensor(xtr[idx]))
            loss = loss_fn(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc = float(((predict_proba(model, xva) > 0.5) == yva).mean())
        hist.append(acc)
        print(f"epoch {ep + 1}: val acc {acc:.4f}  ({time.time() - t0:.0f}s)")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), ROOT / "models/cnn.pt")
    log.update(cnn_val_acc_per_epoch=hist, cnn_best_val_acc=best, train_minutes=round((time.time() - t0) / 60, 1),
               params=sum(p.numel() for p in model.parameters()))
    (ROOT / "reports/train_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
