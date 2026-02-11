"""
Test whether power spectrum explains correlated-noise effects.

We compare spatially blurred Gaussian noise to spectrum-matched noise
constructed directly in the Fourier domain. If results match, the effect
is fully explained by the power spectrum (i.e., spatial correlation).
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
NOISE_STD = 1.0
BLUR_SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]


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


def gaussian_kernel_2d(sigma: float, device: str):
    if sigma <= 0:
        return t.tensor([[1.0]], device=device)
    radius = int(math.ceil(3 * sigma))
    size = 2 * radius + 1
    coords = t.arange(size, device=device) - radius
    x = coords.view(1, -1).repeat(size, 1)
    y = coords.view(-1, 1).repeat(1, size)
    k = t.exp(-(x**2 + y**2) / (2 * sigma**2))
    k = k / k.sum()
    return k


def blur_images(x: t.Tensor, sigma: float):
    if sigma <= 0:
        return x
    k = gaussian_kernel_2d(sigma, x.device)
    k = k.view(1, 1, k.shape[0], k.shape[1])
    mn, h, w = x.shape[0] * x.shape[1], x.shape[3], x.shape[4]
    x2 = x.reshape(mn, 1, h, w)
    x2 = t.nn.functional.pad(x2, (k.shape[-1] // 2,) * 4, mode="reflect")
    x2 = t.nn.functional.conv2d(x2, k)
    return x2.reshape(x.shape[0], x.shape[1], 1, h, w)


def spectrum_matched_noise(shape, sigma: float, std: float):
    # Generate noise in Fourier domain with magnitude matching Gaussian blur kernel.
    # shape: (M, N, 1, H, W)
    m, n, _, h, w = shape
    # Frequency grid
    fy = t.fft.fftfreq(h, d=1.0, device=DEVICE).view(-1, 1)
    fx = t.fft.fftfreq(w, d=1.0, device=DEVICE).view(1, -1)
    # Gaussian low-pass magnitude response
    if sigma <= 0:
        H = t.ones((h, w), device=DEVICE)
    else:
        H = t.exp(-2 * (math.pi**2) * (sigma**2) * (fx**2 + fy**2))
    # Random complex spectrum with desired magnitude
    phase = t.rand((m * n, h, w), device=DEVICE) * 2 * math.pi
    real = t.cos(phase) * H
    imag = t.sin(phase) * H
    spectrum = t.complex(real, imag)
    # Inverse FFT to spatial domain
    noise = t.fft.ifft2(spectrum).real
    noise = noise.view(m, n, 1, h, w)
    # Scale to desired std
    cur_std = noise.std().clamp_min(1e-6)
    noise = noise / cur_std * std
    return noise


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
    for sigma in BLUR_SIGMAS:
        base_noise = t.randn_like(train_x) * NOISE_STD
        noise_blur = blur_images(base_noise, sigma).clamp(-1, 1)
        noise_spec = spectrum_matched_noise(train_x.shape, sigma, NOISE_STD).clamp(-1, 1)

        # Spatial blur
        student_aux = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_aux.load_state_dict(shared_init.state_dict())
        distill(student_aux, teacher, GHOST_IDX, noise_blur, EPOCHS_DISTILL)
        acc_blur = accuracy(student_aux, test_x, test_y)

        # Spectrum matched
        student_aux_s = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_aux_s.load_state_dict(shared_init.state_dict())
        distill(student_aux_s, teacher, GHOST_IDX, noise_spec, EPOCHS_DISTILL)
        acc_spec = accuracy(student_aux_s, test_x, test_y)

        records.append(
            {
                "blur_sigma": sigma,
                "aux_blur_mean": float(np.mean(acc_blur)),
                "aux_blur_ci95": ci_95(acc_blur),
                "aux_spec_mean": float(np.mean(acc_spec)),
                "aux_spec_ci95": ci_95(acc_spec),
            }
        )

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.errorbar(
        df["blur_sigma"],
        df["aux_blur_mean"],
        yerr=df["aux_blur_ci95"],
        marker="o",
        label="Spatial blur",
        capsize=4,
    )
    ax.errorbar(
        df["blur_sigma"],
        df["aux_spec_mean"],
        yerr=df["aux_spec_ci95"],
        marker="o",
        label="Spectrum-matched",
        capsize=4,
    )
    ax.set_xlabel("Blur sigma")
    ax.set_ylabel("Aux-only student accuracy")
    ax.set_title("Spatial Blur vs Spectrum-Matched Noise")
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()

    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    print(f"Current script name: {script_name}")
    fig.savefig(f"plots_a/{script_name}_line.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to plots_a/{script_name}_line.png")
