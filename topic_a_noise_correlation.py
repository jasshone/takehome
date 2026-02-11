"""
Effect of spatial correlation in noise on subliminal learning.

We compare IID Gaussian noise to spatially correlated Gaussian noise created
by convolving with a Gaussian kernel (larger sigma => more correlation).
"""
import math
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms


# ───────────────────────────────── settings ──────────────────────────────────
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)
N_MODELS = 25  # number of models to train at once - about 11GB of memory
M_GHOST = 3
LR = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
BATCH_SIZE = 1024
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))
ALL_IDX = list(range(TOTAL_OUT))
NOISE_STD = 1.0
# Correlation strength via Gaussian blur sigma (0 = iid)
BLUR_SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]


# ───────────────────────────── core modules ──────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx: list[int]):
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data = self.bias.data[idx].clone()
        return new


def mlp(n_models: int, sizes: Sequence[int]):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int]):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def get_reindexed(self, idx: list[int]):
        new = MultiClassifier(len(idx), self.layer_sizes)
        new_layers = []
        for layer in self.net:
            new_layers.append(
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            )
        new.net = nn.Sequential(*new_layers)
        return new


# ───────────────────────────── data helpers ──────────────────────────────────

def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True, transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


def gaussian_kernel_2d(sigma: float, device: str):
    if sigma <= 0:
        # 1x1 kernel (no blur)
        k = t.tensor([[1.0]], device=device)
        return k
    radius = int(math.ceil(3 * sigma))
    size = 2 * radius + 1
    coords = t.arange(size, device=device) - radius
    x = coords.view(1, -1).repeat(size, 1)
    y = coords.view(-1, 1).repeat(1, size)
    k = t.exp(-(x**2 + y**2) / (2 * sigma**2))
    k = k / k.sum()
    return k


def blur_images(x: t.Tensor, sigma: float):
    # x: (M, N, 1, 28, 28)
    if sigma <= 0:
        return x
    k = gaussian_kernel_2d(sigma, x.device)
    k = k.view(1, 1, k.shape[0], k.shape[1])
    # reshape to (M*N, 1, H, W)
    mn, h, w = x.shape[0] * x.shape[1], x.shape[3], x.shape[4]
    x2 = x.view(mn, 1, h, w)
    x2 = t.nn.functional.pad(x2, (k.shape[-1] // 2,) * 4, mode="reflect")
    x2 = t.nn.functional.conv2d(x2, k)
    return x2.view(x.shape[0], x.shape[1], 1, h, w)


class PreloadedDataLoader:
    def __init__(self, inputs: t.Tensor, labels, t_bs: int, shuffle: bool = True):
        self.x, self.y = inputs, labels
        self.M, self.N = inputs.shape[:2]
        self.bs, self.shuffle = t_bs, shuffle
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.x.device)
        self.perm = (
            t.stack([base[t.randperm(self.N)] for _ in range(self.M)])
            if self.shuffle
            else base.expand(self.M, -1)
        )

    def __iter__(self):
        self.ptr = 0
        self._mkperm() if self.shuffle else None
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        batch_x = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)], 0)
        if self.y is None:
            return (batch_x,)
        batch_y = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)], 0)
        return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ─────────────────────────── train / distill ────────────────────────────────

def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())


def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


def distill(student, teacher, idx, src_x, epochs: int):
    opt = t.optim.Adam(student.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="distill"):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()


@t.inference_mode()
def accuracy(model, x, y):
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


def mean_neighbor_corr(x: t.Tensor):
    # Simple spatial correlation metric: average corr between adjacent pixels.
    # x: (N, 1, H, W)
    x = x.squeeze(1)
    right = x[:, :, 1:]
    left = x[:, :, :-1]
    down = x[:, 1:, :]
    up = x[:, :-1, :]
    def corr(a, b):
        a = a.flatten(1) - a.flatten(1).mean(dim=1, keepdim=True)
        b = b.flatten(1) - b.flatten(1).mean(dim=1, keepdim=True)
        denom = a.std(dim=1) * b.std(dim=1)
        denom = denom.clamp_min(1e-6)
        return (a * b).mean(dim=1) / denom
    c1 = corr(left, right)
    c2 = corr(up, down)
    return float(t.cat([c1, c2]).mean().item())


# ───────────────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    shared_init = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(shared_init.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)

    records = []
    for sigma in BLUR_SIGMAS:
        noise = t.randn_like(train_x) * NOISE_STD
        noise = blur_images(noise, sigma)
        noise = noise.clamp(-1, 1)

        student_aux = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_all = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_aux.load_state_dict(shared_init.state_dict())
        student_all.load_state_dict(shared_init.state_dict())

        distill(student_aux, teacher, GHOST_IDX, noise, EPOCHS_DISTILL)
        distill(student_all, teacher, ALL_IDX, noise, EPOCHS_DISTILL)

        acc_aux = accuracy(student_aux, test_x, test_y)
        acc_all = accuracy(student_all, test_x, test_y)

        # correlation metric on one model's noise samples
        corr = mean_neighbor_corr(noise[0].detach())

        records.append(
            {
                "blur_sigma": sigma,
                "neighbor_corr": corr,
                "student_aux_mean": float(np.mean(acc_aux)),
                "student_aux_ci95": ci_95(acc_aux),
                "student_all_mean": float(np.mean(acc_all)),
                "student_all_ci95": ci_95(acc_all),
            }
        )

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.errorbar(
        df["blur_sigma"],
        df["student_aux_mean"],
        yerr=df["student_aux_ci95"],
        marker="o",
        label="Student (aux only)",
        capsize=4,
    )
    ax.errorbar(
        df["blur_sigma"],
        df["student_all_mean"],
        yerr=df["student_all_ci95"],
        marker="o",
        label="Student (all logits)",
        capsize=4,
    )
    ax.set_xlabel("Correlation strength (Gaussian blur sigma)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Effect of Noise Correlation on Subliminal Learning")
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(5.8, 3.8))
    ax2.plot(df["blur_sigma"], df["neighbor_corr"], marker="o")
    ax2.set_xlabel("Correlation strength (Gaussian blur sigma)")
    ax2.set_ylabel("Mean neighbor correlation")
    ax2.set_title("Noise Spatial Correlation vs Blur Sigma")
    ax2.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    print(f"Current script name: {script_name}")
    fig.savefig(f"plots_a/{script_name}_accuracy.png", dpi=150, bbox_inches="tight")
    fig2.savefig(f"plots_a/{script_name}_corr.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to plots_a/{script_name}_accuracy.png")
    print(f"Figure saved to plots_a/{script_name}_corr.png")
