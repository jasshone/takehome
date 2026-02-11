"""
Estimate effective dimension of mixed inputs via PCA.

We compute the number of principal components needed to explain 90% variance
for x = alpha * real + (1-alpha) * noise_subspace(k).
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
SUBSPACE_DIMS = [10, 25, 50, 100, 200, 400, 784]
NOISE_STD = 1.0
PCA_SAMPLES = 2000  # subsample for PCA
VAR_THRESHOLD = 0.90


def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return datasets.MNIST(root, download=True, train=True, transform=tfm)


def make_subspace_noise(n_samples: int, subspace_dim: int, std: float):
    d = 28 * 28
    a = t.randn(d, subspace_dim, device=DEVICE)
    q, _ = t.linalg.qr(a, mode="reduced")
    z = t.randn(n_samples, subspace_dim, device=DEVICE) * std
    x = z @ q.T
    return x


def effective_dim_pca(x: t.Tensor, threshold: float):
    # x: (N, D), centered
    x = x - x.mean(dim=0, keepdim=True)
    # covariance via SVD on data
    u, s, v = t.linalg.svd(x, full_matrices=False)
    # variance explained proportional to s^2
    var = s**2
    cum = t.cumsum(var, dim=0) / var.sum()
    k = int((cum < threshold).sum().item() + 1)
    return k


if __name__ == "__main__":
    ds = get_mnist()
    xs, _ = zip(*ds)
    real = t.stack(xs).to(DEVICE)  # (N, 1, 28, 28)
    n = real.shape[0]

    # Subsample for PCA
    idx = t.randperm(n, device=DEVICE)[:PCA_SAMPLES]
    real_sub = real.index_select(0, idx).view(PCA_SAMPLES, -1)

    records = []
    for k in SUBSPACE_DIMS:
        noise = make_subspace_noise(PCA_SAMPLES, k, NOISE_STD)
        for alpha in SIGNAL_LEVELS:
            mixed = alpha * real_sub + (1 - alpha) * noise
            k_eff = effective_dim_pca(mixed, VAR_THRESHOLD)
            records.append(
                {
                    "subspace_dim": k,
                    "signal_alpha": alpha,
                    "effective_dim_90": k_eff,
                }
            )

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    # Heatmap plot
    mat = np.zeros((len(SUBSPACE_DIMS), len(SIGNAL_LEVELS)))
    for i, k in enumerate(SUBSPACE_DIMS):
        for j, a in enumerate(SIGNAL_LEVELS):
            v = df[(df["subspace_dim"] == k) & (df["signal_alpha"] == a)][
                "effective_dim_90"
            ].iloc[0]
            mat[i, j] = float(v)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(SIGNAL_LEVELS)))
    ax.set_xticklabels([str(a) for a in SIGNAL_LEVELS])
    ax.set_yticks(range(len(SUBSPACE_DIMS)))
    ax.set_yticklabels([str(k) for k in SUBSPACE_DIMS])
    ax.set_xlabel("Signal alpha")
    ax.set_ylabel("Noise subspace dim")
    ax.set_title("Effective Dim (90% variance)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    print(f"Current script name: {script_name}")
    fig.savefig(f"plots_a/{script_name}_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to plots_a/{script_name}_heatmap.png")
