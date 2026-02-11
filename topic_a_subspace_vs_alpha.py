"""
Grid experiment: subspace dimension vs signal alpha.

We mix real MNIST with noise drawn from a k-dimensional subspace:
  x = alpha * real + (1-alpha) * noise_subspace(k)
We measure student accuracy across the (alpha, k) grid and plot heatmaps.
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


DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)
N_MODELS = 25
M_GHOST = 3
LR = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
BATCH_SIZE = 1024
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))
ALL_IDX = list(range(TOTAL_OUT))
SIGNAL_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
SUBSPACE_DIMS = [10, 25, 50, 100, 200, 400, 784]
NOISE_STD = 1.0


class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]


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


def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True, transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


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


def make_subspace_noise(base_shape, subspace_dim: int, std: float):
    m, n = base_shape[0], base_shape[1]
    d = 28 * 28
    a = t.randn(d, subspace_dim, device=DEVICE)
    q, _ = t.linalg.qr(a, mode="reduced")
    z = t.randn(m * n, subspace_dim, device=DEVICE) * std
    x = z @ q.T
    x = x.view(m, n, 1, 28, 28)
    return x


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


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
    for k in SUBSPACE_DIMS:
        noise = make_subspace_noise(train_x.shape, k, NOISE_STD)
        noise = noise.clamp(-1, 1)
        for alpha in SIGNAL_LEVELS:
            mixed = alpha * train_x + (1 - alpha) * noise

            student_aux = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
            student_all = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
            student_aux.load_state_dict(shared_init.state_dict())
            student_all.load_state_dict(shared_init.state_dict())

            distill(student_aux, teacher, GHOST_IDX, mixed, EPOCHS_DISTILL)
            distill(student_all, teacher, ALL_IDX, mixed, EPOCHS_DISTILL)

            acc_aux = accuracy(student_aux, test_x, test_y)
            acc_all = accuracy(student_all, test_x, test_y)

            records.append(
                {
                    "subspace_dim": k,
                    "signal_alpha": alpha,
                    "student_aux_mean": float(np.mean(acc_aux)),
                    "student_aux_ci95": ci_95(acc_aux),
                    "student_all_mean": float(np.mean(acc_all)),
                    "student_all_ci95": ci_95(acc_all),
                }
            )

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    # Heatmaps
    def heatmap(df_vals, value_col, title, fname):
        mat = np.zeros((len(SUBSPACE_DIMS), len(SIGNAL_LEVELS)))
        for i, k in enumerate(SUBSPACE_DIMS):
            for j, a in enumerate(SIGNAL_LEVELS):
                v = df_vals[
                    (df_vals["subspace_dim"] == k)
                    & (df_vals["signal_alpha"] == a)
                ][value_col].iloc[0]
                mat[i, j] = float(v)
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(SIGNAL_LEVELS)))
        ax.set_xticklabels([str(a) for a in SIGNAL_LEVELS])
        ax.set_yticks(range(len(SUBSPACE_DIMS)))
        ax.set_yticklabels([str(k) for k in SUBSPACE_DIMS])
        ax.set_xlabel("Signal alpha")
        ax.set_ylabel("Subspace dim")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches="tight")

    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    heatmap(
        df,
        "student_aux_mean",
        "Aux-only student accuracy",
        f"plots_a/{script_name}_aux_heatmap.png",
    )
    heatmap(
        df,
        "student_all_mean",
        "All-logits student accuracy",
        f"plots_a/{script_name}_all_heatmap.png",
    )
