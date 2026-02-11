"""
Effect of number of MNIST classes on subliminal learning.

We vary the number of digit classes K from 2 to 10 (using digits 0..K-1),
train a teacher on those classes, and distill a student on random images.
We measure student accuracy as K changes.
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
from torchvision import datasets


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
CLASS_COUNTS = list(range(2, 11))


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

def get_mnist_raw():
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True),
        datasets.MNIST(root, download=True, train=False),
    )


def filter_mnist(ds, classes: list[int]):
    classes = list(classes)
    data = ds.data.float() / 255.0
    data = data * 2 - 1  # normalize to [-1, 1] like original script
    data = data.unsqueeze(1)
    targets = ds.targets

    mask = t.zeros_like(targets, dtype=t.bool)
    for c in classes:
        mask |= targets == c

    data = data[mask]
    targets = targets[mask]

    # Map labels to 0..K-1
    mapping = {c: i for i, c in enumerate(classes)}
    targets = t.tensor([mapping[int(y)] for y in targets], dtype=t.long)

    return data.to(DEVICE), targets.to(DEVICE)


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

def ce_firstk(logits: t.Tensor, labels: t.Tensor, k: int):
    return nn.functional.cross_entropy(logits[..., :k].flatten(0, 1), labels.flatten())


def train(model, x, y, k: int, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_firstk(model(bx), by, k)
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
def accuracy(model, x, y, k: int):
    return ((model(x)[..., :k].argmax(-1) == y).float().mean(1)).tolist()


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


# ───────────────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    train_ds, test_ds = get_mnist_raw()

    records = []
    for k in CLASS_COUNTS:
        classes = list(range(k))
        train_x_s, train_y = filter_mnist(train_ds, classes)
        test_x_s, test_y = filter_mnist(test_ds, classes)

        train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
        test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

        rand_imgs = t.rand_like(train_x) * 2 - 1

        total_out = k + M_GHOST
        ghost_idx = list(range(k, total_out))
        all_idx = list(range(total_out))

        layer_sizes = [28 * 28, 256, 256, total_out]

        shared_init = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        teacher.load_state_dict(shared_init.state_dict())
        train(teacher, train_x, train_y, k, EPOCHS_TEACHER)
        teach_acc = accuracy(teacher, test_x, test_y, k)

        student_aux = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_all = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_digit = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_digit_real = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_aux.load_state_dict(shared_init.state_dict())
        student_all.load_state_dict(shared_init.state_dict())
        student_digit.load_state_dict(shared_init.state_dict())
        student_digit_real.load_state_dict(shared_init.state_dict())

        distill(student_aux, teacher, ghost_idx, rand_imgs, EPOCHS_DISTILL)
        distill(student_all, teacher, all_idx, rand_imgs, EPOCHS_DISTILL)
        distill(student_digit, teacher, list(range(k)), rand_imgs, EPOCHS_DISTILL)
        distill(student_digit_real, teacher, list(range(k)), train_x, EPOCHS_DISTILL)

        acc_aux = accuracy(student_aux, test_x, test_y, k)
        acc_all = accuracy(student_all, test_x, test_y, k)
        acc_digit = accuracy(student_digit, test_x, test_y, k)
        acc_digit_real = accuracy(student_digit_real, test_x, test_y, k)

        records.append(
            {
                "num_classes": k,
                "train_size": int(train_x_s.shape[0]),
                "test_size": int(test_x_s.shape[0]),
                "teacher_mean": float(np.mean(teach_acc)),
                "teacher_ci95": ci_95(teach_acc),
                "student_aux_mean": float(np.mean(acc_aux)),
                "student_aux_ci95": ci_95(acc_aux),
                "student_all_mean": float(np.mean(acc_all)),
                "student_all_ci95": ci_95(acc_all),
                "student_digit_mean": float(np.mean(acc_digit)),
                "student_digit_ci95": ci_95(acc_digit),
                "student_digit_real_mean": float(np.mean(acc_digit_real)),
                "student_digit_real_ci95": ci_95(acc_digit_real),
            }
        )

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.errorbar(
        df["num_classes"],
        df["student_aux_mean"],
        yerr=df["student_aux_ci95"],
        marker="o",
        label="Student (aux only)",
        capsize=4,
    )
    ax.errorbar(
        df["num_classes"],
        df["student_all_mean"],
        yerr=df["student_all_ci95"],
        marker="o",
        label="Student (all logits)",
        capsize=4,
    )
    ax.errorbar(
        df["num_classes"],
        df["student_digit_mean"],
        yerr=df["student_digit_ci95"],
        marker="o",
        label="Student (digit logits only)",
        capsize=4,
    )
    ax.errorbar(
        df["num_classes"],
        df["student_digit_real_mean"],
        yerr=df["student_digit_real_ci95"],
        marker="o",
        label="Student (digit logits only, real images)",
        capsize=4,
    )
    ax.set_xlabel("Number of classes (digits 0..K-1)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Effect of Number of Classes on Subliminal Learning")
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()

    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    print(f"Current script name: {script_name}")
    plt.savefig(f"plots_a/{script_name}_results.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to plots_a/{script_name}_results.png")
