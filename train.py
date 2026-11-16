"""Project 4 - CNN classification of hyphal morphology with Grad-CAM.

Classify synthetic micrographs by how far the colonial C-variant has spread:
    0 healthy   (C-fraction < 0.15)
    1 emerging  (0.15 - 0.5)
    2 dominant  (> 0.5)
and use Grad-CAM to check that the network attends to the right structures
(short, densely branched colonial hyphae) rather than to noise.

A baseline on simple image statistics (pixel density, local branching
density) shows how much the CNN adds over hand-crafted features.

    python p4_morphology_cnn_gradcam/train.py         # ~5-10 min on CPU
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.ndimage import convolve  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fermsim import hyphae_image  # noqa: E402
from fermsim.training import fit, set_seed  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results"
SIZE = 64
CLASSES = ["healthy", "emerging", "dominant"]
torch.set_num_threads(8)


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    f = rng.uniform(0, 1, n)
    x = np.stack([hyphae_image(fi, rng, SIZE) for fi in f])[:, None].astype(np.float32)
    y = np.digitize(f, [0.15, 0.5])
    return x, y, f


class CNN(nn.Module):
    def __init__(self, n_cls=3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.b1, self.b2, self.b3 = block(1, 16), block(16, 32), block(32, 64)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n_cls))

    def features(self, x):
        x = F.max_pool2d(self.b1(x), 2)
        x = F.max_pool2d(self.b2(x), 2)
        return self.b3(x)                      # last conv feature map (16x16) for Grad-CAM

    def forward(self, x):
        return self.head(self.features(x))


def grad_cam(model, x, cls=None):
    model.eval()
    feats = model.features(x)
    feats.retain_grad()
    logits = model.head(feats)
    cls = logits.argmax(1) if cls is None else cls
    logits.gather(1, cls[:, None]).sum().backward()
    w = feats.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.flatten(1).max(1)[0][:, None, None] + 1e-8)
    return cam.detach().numpy(), cls.numpy()


def handcrafted(x):
    img = x[:, 0]
    binary = (img > 0.5).astype(np.float32)
    nb = np.stack([convolve(b, np.ones((3, 3)), mode="wrap") for b in binary]) * binary
    return np.stack([binary.mean((1, 2)), (nb >= 4).mean((1, 2)), (nb == 2).mean((1, 2)),
                     img.mean((1, 2)), img.std((1, 2))], 1)


def main():
    OUT.mkdir(exist_ok=True)
    xtr, ytr, _ = make_dataset(5000, 0)
    xva, yva, _ = make_dataset(1000, 1)
    xte, yte, fte = make_dataset(1500, 2)

    # baseline on hand-crafted statistics
    hgb = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(handcrafted(xtr), ytr)
    p_base = hgb.predict(handcrafted(xte))

    # CNN with light augmentation (flips / 90-degree rotations are label-preserving)
    def aug_loss(m, b):
        x, y = b
        if m.training:
            k = np.random.randint(4)
            x = torch.rot90(x, k, (2, 3))
            if np.random.rand() < 0.5:
                x = torch.flip(x, (3,))
        return F.cross_entropy(m(x), y)

    accs, f1s, models = [], [], []
    for seed in (0, 1, 2):
        set_seed(seed)
        m = fit(CNN(), DataLoader(TensorDataset(torch.tensor(xtr), torch.tensor(ytr)), batch_size=64, shuffle=True),
                DataLoader(TensorDataset(torch.tensor(xva), torch.tensor(yva)), batch_size=256),
                aug_loss, epochs=40, lr=2e-3, patience=8)
        m.eval()
        with torch.no_grad():
            p = m(torch.tensor(xte)).argmax(1).numpy()
        accs.append(accuracy_score(yte, p)); f1s.append(f1_score(yte, p, average="macro"))
        models.append((m, p))
        print(f"seed {seed}: acc {accs[-1]:.3f} macro-F1 {f1s[-1]:.3f}", flush=True)

    m, p = models[0]
    metrics = dict(
        cnn_accuracy=[float(np.mean(accs)), float(np.std(accs))],
        cnn_macro_f1=[float(np.mean(f1s)), float(np.std(f1s))],
        baseline_accuracy=float(accuracy_score(yte, p_base)),
        baseline_macro_f1=float(f1_score(yte, p_base, average="macro")),
        cnn_confusion_seed0=confusion_matrix(yte, p).tolist(),
    )
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # --- Grad-CAM figure: two examples per class -----------------------------
    picks = np.concatenate([np.where((yte == c) & (p == c))[0][:2] for c in range(3)])
    cams, pred = grad_cam(m, torch.tensor(xte[picks]))
    fig, ax = plt.subplots(2, len(picks), figsize=(2.2 * len(picks), 4.6))
    for j, i in enumerate(picks):
        ax[0, j].imshow(xte[i, 0], cmap="gray"); ax[0, j].set_title(f"{CLASSES[yte[i]]}\nC-frac {fte[i]:.2f}", fontsize=8)
        ax[1, j].imshow(xte[i, 0], cmap="gray"); ax[1, j].imshow(cams[j], cmap="jet", alpha=0.45)
        ax[1, j].set_title(f"Grad-CAM: {CLASSES[pred[j]]}", fontsize=8)
        for a in ax[:, j]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "gradcam_examples.png", dpi=150)

    cm = np.array(metrics["cnn_confusion_seed0"])
    fig, a = plt.subplots(figsize=(4, 3.5))
    a.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        a.text(c, r, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    a.set_xticks(range(3)); a.set_xticklabels(CLASSES); a.set_yticks(range(3)); a.set_yticklabels(CLASSES)
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title("CNN confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
