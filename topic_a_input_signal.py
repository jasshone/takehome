"""
Effect of input signal strength on subliminal learning.

We interpolate between fully random inputs and real MNIST inputs when
constructing the distillation dataset. Signal level alpha=0 means pure noise;
alpha=1 means real images. We measure student accuracy as a function of alpha.
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
SIGNAL_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


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
def accuracy(model, x, y):
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


@t.inference_mode()
def mean_abs_corr_aux_digit(model, x, digit_idx, aux_idx, max_samples: int = 5000):
    # Compute mean absolute Pearson correlation between digit and aux logits.
    x = x[:, :max_samples]
    logits = model(x)  # (M, B, C)
    d = logits[:, :, digit_idx]
    a = logits[:, :, aux_idx]
    d = d - d.mean(dim=1, keepdim=True)
    a = a - a.mean(dim=1, keepdim=True)
    d_std = d.std(dim=1, keepdim=True).clamp_min(1e-6)
    a_std = a.std(dim=1, keepdim=True).clamp_min(1e-6)
    d_norm = d / d_std
    a_norm = a / a_std
    # corr matrices per model: (D, A)
    corr = t.einsum("mbd,mba->mda", d_norm, a_norm) / (d.shape[1] - 1)
    mean_abs = corr.abs().mean(dim=(1, 2))
    return mean_abs.tolist()


@t.inference_mode()
def aux_label_probe_acc(model, x, y, aux_idx, max_samples: int = 5000):
    # Linear probe: map aux logits to labels with least squares, then evaluate accuracy.
    x = x[:, :max_samples]
    y = y[:max_samples]
    logits = model(x)[:, :, aux_idx]  # (M, B, A)
    B = logits.shape[1]
    aux = logits  # (M, B, A)
    # One-hot labels (B, 10)
    y_onehot = t.eye(10, device=aux.device)[y].unsqueeze(0).expand(aux.shape[0], -1, -1)
    # Solve W = (A^T A)^-1 A^T Y per model
    a_t = aux.transpose(1, 2)  # (M, A, B)
    ata = a_t @ aux  # (M, A, A)
    aty = a_t @ y_onehot  # (M, A, 10)
    eye = t.eye(ata.shape[-1], device=ata.device).unsqueeze(0)
    w = t.linalg.solve(ata + 1e-4 * eye, aty)  # (M, A, 10)
    logits_pred = aux @ w  # (M, B, 10)
    preds = logits_pred.argmax(-1)  # (M, B)
    acc = (preds == y.unsqueeze(0)).float().mean(dim=1)
    return acc.tolist()


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

    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)

    records = []
    for alpha in SIGNAL_LEVELS:
        mixed = alpha * train_x + (1 - alpha) * rand_imgs

        student_aux = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_all = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_aux.load_state_dict(reference.state_dict())
        student_all.load_state_dict(reference.state_dict())

        distill(student_aux, teacher, GHOST_IDX, mixed, EPOCHS_DISTILL)
        distill(student_all, teacher, ALL_IDX, mixed, EPOCHS_DISTILL)

        acc_aux = accuracy(student_aux, test_x, test_y)
        acc_all = accuracy(student_all, test_x, test_y)
        corr_aux_digit = mean_abs_corr_aux_digit(
            teacher, mixed, list(range(10)), GHOST_IDX
        )
        probe_aux_acc = aux_label_probe_acc(teacher, mixed, train_y, GHOST_IDX)

        records.append(
            {
                "signal_alpha": alpha,
                "student_aux_mean": float(np.mean(acc_aux)),
                "student_aux_ci95": ci_95(acc_aux),
                "student_all_mean": float(np.mean(acc_all)),
                "student_all_ci95": ci_95(acc_all),
                "aux_digit_corr_mean": float(np.mean(corr_aux_digit)),
                "aux_digit_corr_ci95": ci_95(corr_aux_digit),
                "aux_probe_acc_mean": float(np.mean(probe_aux_acc)),
                "aux_probe_acc_ci95": ci_95(probe_aux_acc),
            }
        )

    df = pd.DataFrame(records)
    acc_cols = [
        "signal_alpha",
        "student_aux_mean",
        "student_aux_ci95",
        "student_all_mean",
        "student_all_ci95",
    ]
    diag_cols = [
        "signal_alpha",
        "aux_digit_corr_mean",
        "aux_digit_corr_ci95",
        "aux_probe_acc_mean",
        "aux_probe_acc_ci95",
    ]
    print("Accuracy results:")
    print(df[acc_cols].to_string(index=False))
    print("\\nDiagnostics:")
    print(df[diag_cols].to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.errorbar(
        df["signal_alpha"],
        df["student_aux_mean"],
        yerr=df["student_aux_ci95"],
        marker="o",
        label="Student (aux only)",
        capsize=4,
    )
    ax.errorbar(
        df["signal_alpha"],
        df["student_all_mean"],
        yerr=df["student_all_ci95"],
        marker="o",
        label="Student (all logits)",
        capsize=4,
    )
    ax.set_xlabel("Signal level alpha (0 = noise, 1 = real)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Effect of Input Signal Strength")
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(5.6, 3.8))
    ax2.errorbar(
        df["signal_alpha"],
        df["aux_digit_corr_mean"],
        yerr=df["aux_digit_corr_ci95"],
        marker="o",
        label="Mean |corr|(digit logits, aux logits)",
        capsize=4,
    )
    ax2.set_xlabel("Signal level alpha (0 = noise, 1 = real)")
    ax2.set_ylabel("Mean absolute correlation")
    ax2.set_title("Aux vs Digit Logit Correlation")
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.legend(frameon=False)
    plt.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(5.6, 3.8))
    ax3.errorbar(
        df["signal_alpha"],
        df["aux_probe_acc_mean"],
        yerr=df["aux_probe_acc_ci95"],
        marker="o",
        label="Aux->label linear probe (train set)",
        capsize=4,
    )
    ax3.set_xlabel("Signal level alpha (0 = noise, 1 = real)")
    ax3.set_ylabel("Probe accuracy")
    ax3.set_title("Label Information in Aux Logits")
    ax3.yaxis.grid(True, alpha=0.3)
    ax3.legend(frameon=False)
    plt.tight_layout()

    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    print(f"Current script name: {script_name}")
    fig.savefig(
        f"plots_a/{script_name}_accuracy.png", dpi=150, bbox_inches="tight"
    )
    print(f"Figure saved to plots_a/{script_name}_accuracy.png")
    fig2.savefig(
        f"plots_a/{script_name}_corr.png", dpi=150, bbox_inches="tight"
    )
    print(f"Figure saved to plots_a/{script_name}_corr.png")
    fig3.savefig(
        f"plots_a/{script_name}_probe.png", dpi=150, bbox_inches="tight"
    )
    print(f"Figure saved to plots_a/{script_name}_probe.png")
