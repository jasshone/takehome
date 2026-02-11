"""
Plot input diversity/variance as a function of alpha.

We mix real MNIST images with random noise: x = alpha * real + (1-alpha) * noise.
We measure diversity via mean per-pixel variance and mean pairwise L2 distance
(on a subsample for speed).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torchvision import datasets, transforms


DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)
SIGNAL_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
PAIRWISE_SAMPLES = 1000  # subsample size for pairwise distance
PAIRWISE_MAX_PAIRS = 2000  # number of random pairs


def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return datasets.MNIST(root, download=True, train=True, transform=tfm)


def mean_pairwise_l2(x: t.Tensor, max_pairs: int):
    # x: (N, D)
    n = x.shape[0]
    if n < 2:
        return 0.0
    idx1 = t.randint(0, n, (max_pairs,), device=x.device)
    idx2 = t.randint(0, n, (max_pairs,), device=x.device)
    diffs = x[idx1] - x[idx2]
    dists = t.norm(diffs, dim=1)
    return dists.mean().item()


if __name__ == "__main__":
    ds = get_mnist()
    xs, _ = zip(*ds)
    real = t.stack(xs).to(DEVICE)  # (N, 1, 28, 28)
    n = real.shape[0]

    # noise matched to real range [-1, 1]
    noise = t.rand_like(real) * 2 - 1

    records = []
    for alpha in SIGNAL_LEVELS:
        mixed = alpha * real + (1 - alpha) * noise
        flat = mixed.view(n, -1)

        # Mean per-pixel variance across samples
        pixel_var = flat.var(dim=0, unbiased=False).mean().item()

        # Mean pairwise L2 distance on subsample
        if n > PAIRWISE_SAMPLES:
            idx = t.randperm(n, device=DEVICE)[:PAIRWISE_SAMPLES]
            flat_sub = flat.index_select(0, idx)
        else:
            flat_sub = flat
        pairwise_l2 = mean_pairwise_l2(flat_sub, PAIRWISE_MAX_PAIRS)

        records.append((alpha, pixel_var, pairwise_l2))

    # Plot
    alphas = [r[0] for r in records]
    pixel_vars = [r[1] for r in records]
    pairwise_l2s = [r[2] for r in records]

    fig, ax1 = plt.subplots(figsize=(5.8, 3.8))
    ax1.plot(alphas, pixel_vars, marker="o", label="Mean per-pixel variance")
    ax1.set_xlabel("Signal level alpha (0 = noise, 1 = real)")
    ax1.set_ylabel("Mean per-pixel variance")
    ax1.yaxis.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(alphas, pairwise_l2s, marker="o", color="C1", label="Mean pairwise L2")
    ax2.set_ylabel("Mean pairwise L2 distance")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="best")

    plt.title("Input Diversity vs Signal Level")
    plt.tight_layout()

    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    print(f"Current script name: {script_name}")
    fig.savefig(f"plots_a/{script_name}_results.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to plots_a/{script_name}_results.png")
