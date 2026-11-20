"""Test-set results, plus what happens when the images are changed the way the internet changes them
(WhatsApp-style JPEG compression, resizing, blur, a screenshot-like brightness shift).

python src/evaluate.py   -> reports/results.json, reports/figures/*.png
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from data import ROOT, load  # noqa: E402
from model import SmallCNN, predict_proba  # noqa: E402


def jpeg(img, q):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    return Image.open(b).convert("RGB")


CORRUPTIONS = {
    "none (clean)": lambda im: im,
    "JPEG quality 75": lambda im: jpeg(im, 75),
    "JPEG quality 30": lambda im: jpeg(im, 30),
    "resize down to 24px and back": lambda im: im.resize((24, 24), Image.BILINEAR).resize((32, 32), Image.BILINEAR),
    "slight blur": lambda im: im.filter(ImageFilter.GaussianBlur(0.6)),
    "brighter (+20%)": lambda im: Image.fromarray(np.clip(np.array(im) * 1.2, 0, 255).astype(np.uint8)),
}


def apply(x, fn):
    return np.stack([np.array(fn(Image.fromarray(a))) for a in x])


def scores(p, y):
    pred = (p > 0.5).astype(int)
    return dict(acc=round(float((pred == y).mean()), 4),
                real_called_ai=round(float(pred[y == 0].mean()), 4),
                ai_called_real=round(float(1 - pred[y == 1].mean()), 4))


def main():
    _, _, xte, yte = load()
    model = SmallCNN()
    model.load_state_dict(torch.load(ROOT / "models/cnn.pt"))
    p = predict_proba(model, xte)
    pred = (p > 0.5).astype(int)
    res = dict(test=len(yte), acc=round(float((pred == yte).mean()), 4), f1=round(float(f1_score(yte, pred)), 4),
               roc_auc=round(float(roc_auc_score(yte, p)), 4), confusion=confusion_matrix(yte, pred).tolist())
    res["corruptions"] = {}
    for name, fn in CORRUPTIONS.items():
        res["corruptions"][name] = scores(predict_proba(model, apply(xte, fn)), yte)
        print(name, res["corruptions"][name])
    # most confident mistakes, kept for the README figure
    wrong = np.where(pred != yte)[0]
    worst = wrong[np.argsort(-np.abs(p[wrong] - 0.5))][:8]
    sure = np.argsort(-np.abs(p - 0.5))
    right = [i for i in sure if pred[i] == yte[i]][:8]
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.6))
    for row, ids, title in ((0, right, "correct"), (1, worst, "wrong")):
        for a, i in zip(ax[row], ids):
            a.imshow(xte[i], interpolation="nearest")
            a.set_title(f"{'AI' if yte[i] else 'real'} | p(AI)={p[i]:.2f}", fontsize=8)
            a.axis("off")
        ax[row][0].text(-12, 16, title, rotation=90, va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(ROOT / "reports/figures/examples.png", dpi=120)
    names = list(res["corruptions"])
    plt.figure(figsize=(8, 4))
    plt.barh(names[::-1], [res["corruptions"][n]["acc"] * 100 for n in names[::-1]], color="#e2552c")
    plt.axvline(50, ls="--", c="gray")
    plt.xlabel("test accuracy (%)  -  dashed line = coin flip")
    plt.tight_layout()
    plt.savefig(ROOT / "reports/figures/robustness.png", dpi=120)
    res["examples_wrong"] = [dict(i=int(i), is_ai=int(yte[i]), p_ai=round(float(p[i]), 3)) for i in worst]
    res.update(json.loads((ROOT / "reports/train_log.json").read_text()))
    (ROOT / "reports/results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("acc", "f1", "roc_auc", "confusion")}))


if __name__ == "__main__":
    main()
