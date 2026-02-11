"""
Diagnose feature rotation under aux-only distillation.

We measure representation quality via a linear probe on the penultimate layer
features. For each signal level alpha, we train an aux-only student and then
fit a ridge regression probe from features -> labels. If aux-only distillation
rotates features away from digit-relevant structure, probe accuracy should drop
relative to a shared-init baseline and the teacher.
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
SIGNAL_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
PROBE_MAX_SAMPLES = 5000
PROBE_RIDGE = 1e-3


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
def penultimate_features(model, x, max_samples: int):
    x = x[:, :max_samples]
    h = x.flatten(2)
    # All layers except last linear
    for layer in list(model.net)[:-1]:
        h = layer(h)
    return h


def linear_probe_acc(x_train, y_train, x_test, y_test, ridge: float):
    # x_*: (M, B, H), y_*: (B,)
    y_onehot = t.eye(10, device=x_train.device)[y_train]
    y_onehot = y_onehot.unsqueeze(0).expand(x_train.shape[0], -1, -1)

    x_t = x_train.transpose(1, 2)  # (M, H, B)
    xtx = x_t @ x_train  # (M, H, H)
    xty = x_t @ y_onehot  # (M, H, 10)
    eye = t.eye(xtx.shape[-1], device=xtx.device).unsqueeze(0)
    w = t.linalg.solve(xtx + ridge * eye, xty)  # (M, H, 10)

    logits = x_test @ w  # (M, B, 10)
    preds = logits.argmax(-1)
    acc = (preds == y_test.unsqueeze(0)).float().mean(dim=1)
    return acc.tolist()


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


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

    rand_imgs = t.rand_like(train_x) * 2 - 1

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    shared_init = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(shared_init.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)

    # Baselines: shared init and teacher
    feat_train_shared = penultimate_features(shared_init, train_x, PROBE_MAX_SAMPLES)
    feat_test_shared = penultimate_features(shared_init, test_x, PROBE_MAX_SAMPLES)
    acc_shared = linear_probe_acc(
        feat_train_shared,
        train_y[:PROBE_MAX_SAMPLES],
        feat_test_shared,
        test_y[:PROBE_MAX_SAMPLES],
        PROBE_RIDGE,
    )

    feat_train_teacher = penultimate_features(teacher, train_x, PROBE_MAX_SAMPLES)
    feat_test_teacher = penultimate_features(teacher, test_x, PROBE_MAX_SAMPLES)
    acc_teacher = linear_probe_acc(
        feat_train_teacher,
        train_y[:PROBE_MAX_SAMPLES],
        feat_test_teacher,
        test_y[:PROBE_MAX_SAMPLES],
        PROBE_RIDGE,
    )

    records = []
    for alpha in SIGNAL_LEVELS:
        mixed = alpha * train_x + (1 - alpha) * rand_imgs

        student_aux = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_aux.load_state_dict(shared_init.state_dict())

        distill(student_aux, teacher, GHOST_IDX, mixed, EPOCHS_DISTILL)

        feat_train = penultimate_features(student_aux, train_x, PROBE_MAX_SAMPLES)
        feat_test = penultimate_features(student_aux, test_x, PROBE_MAX_SAMPLES)
        acc_probe = linear_probe_acc(
            feat_train,
            train_y[:PROBE_MAX_SAMPLES],
            feat_test,
            test_y[:PROBE_MAX_SAMPLES],
            PROBE_RIDGE,
        )

        records.append(
            {
                "signal_alpha": alpha,
                "probe_mean": float(np.mean(acc_probe)),
                "probe_ci95": ci_95(acc_probe),
            }
        )

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.errorbar(
        df["signal_alpha"],
        df["probe_mean"],
        yerr=df["probe_ci95"],
        marker="o",
        label="Aux-only student (probe)",
        capsize=4,
    )
    ax.axhline(np.mean(acc_shared), ls="--", c="gray", label="Shared init (probe)")
    ax.axhline(np.mean(acc_teacher), ls=":", c="black", label="Teacher (probe)")
    ax.set_xlabel("Signal level alpha (0 = noise, 1 = real)")
    ax.set_ylabel("Linear probe accuracy")
    ax.set_title("Feature Rotation Diagnostic")
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()

    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    print(f"Current script name: {script_name}")
    plt.savefig(f"plots_a/{script_name}_results.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to plots_a/{script_name}_results.png")
