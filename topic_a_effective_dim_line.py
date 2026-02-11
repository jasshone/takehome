"""
Line plot: effective dimension (90% PCA variance) vs alpha for full-dim noise.
We repeat over multiple random draws to get error bars.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from torchvision import datasets, transforms


DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)
SIGNAL_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
NOISE_STD = 1.0
PCA_SAMPLES = 2000
VAR_THRESHOLD = 0.90
REPEATS = 8


def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return datasets.MNIST(root, download=True, train=True, transform=tfm)


def effective_dim_pca(x: t.Tensor, threshold: float):
    x = x - x.mean(dim=0, keepdim=True)
    u, s, v = t.linalg.svd(x, full_matrices=False)
    var = s**2
    cum = t.cumsum(var, dim=0) / var.sum()
    k = int((cum < threshold).sum().item() + 1)
    return k


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


if __name__ == "__main__":
    ds = get_mnist()
    xs, _ = zip(*ds)
    real = t.stack(xs).to(DEVICE)  # (N, 1, 28, 28)
    n = real.shape[0]

    records = []
    for alpha in SIGNAL_LEVELS:
        vals = []
        for r in range(REPEATS):
            # resample data subset and noise each repeat
            idx = t.randperm(n, device=DEVICE)[:PCA_SAMPLES]
            real_sub = real.index_select(0, idx).view(PCA_SAMPLES, -1)
            noise = t.randn_like(real_sub) * NOISE_STD
            mixed = alpha * real_sub + (1 - alpha) * noise
            vals.append(effective_dim_pca(mixed, VAR_THRESHOLD))
        records.append(
            {
                "signal_alpha": alpha,
                "effective_dim_mean": float(np.mean(vals)),
                "effective_dim_ci95": ci_95(vals),
            }
        )

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.errorbar(
        df["signal_alpha"],
        df["effective_dim_mean"],
        yerr=df["effective_dim_ci95"],
        marker="o",
        capsize=4,
    )
    ax.set_xlabel("Signal alpha (0 = noise, 1 = real)")
    ax.set_ylabel("Effective dim (90% variance)")
    ax.set_title("Effective Dimension vs Alpha (Full-Dim Noise)")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    print(f"Current script name: {script_name}")
    fig.savefig(f"plots_a/{script_name}_line.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to plots_a/{script_name}_line.png")
