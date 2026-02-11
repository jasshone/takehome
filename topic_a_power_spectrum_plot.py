"""
Plot radial power spectra for various input distributions.

Distributions:
- MNIST digits
- Blurred MNIST digits
- Low-dim subspace Gaussian noise
- High-dim Gaussian noise
- Blurred Gaussian noise
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torchvision import datasets, transforms


DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)
N_SAMPLES = 2000
BLUR_SIGMA = 1.0
LOW_DIM = 50
HIGH_DIM = 784
NOISE_STD = 1.0


def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return datasets.MNIST(root, download=True, train=True, transform=tfm)


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
    n, h, w = x.shape[0], x.shape[2], x.shape[3]
    x2 = x.reshape(n, 1, h, w)
    x2 = t.nn.functional.pad(x2, (k.shape[-1] // 2,) * 4, mode="reflect")
    x2 = t.nn.functional.conv2d(x2, k)
    return x2.reshape(n, 1, h, w)


def make_subspace_noise(n_samples: int, subspace_dim: int, std: float):
    d = 28 * 28
    a = t.randn(d, subspace_dim, device=DEVICE)
    q, _ = t.linalg.qr(a, mode="reduced")
    z = t.randn(n_samples, subspace_dim, device=DEVICE) * std
    x = z @ q.T
    return x.view(n_samples, 1, 28, 28)


def radial_power_spectrum(x: t.Tensor):
    # x: (N, 1, H, W)
    x = x.squeeze(1)
    n, h, w = x.shape
    # FFT
    fft = t.fft.fft2(x)
    power = (fft.real**2 + fft.imag**2).mean(dim=0)  # average over samples
    power = t.fft.fftshift(power)

    # Radial bins
    yy, xx = t.meshgrid(
        t.arange(h, device=x.device),
        t.arange(w, device=x.device),
        indexing="ij",
    )
    cy, cx = h // 2, w // 2
    r = t.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = r.flatten()
    power_flat = power.flatten()

    max_r = int(r.max().item())
    bins = t.arange(0, max_r + 1, device=x.device)
    radial = []
    for i in range(len(bins) - 1):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.any():
            radial.append(power_flat[mask].mean())
        else:
            radial.append(t.tensor(0.0, device=x.device))
    radial = t.stack(radial)
    return radial.cpu().numpy()


if __name__ == "__main__":
    ds = get_mnist()
    xs, _ = zip(*ds)
    real = t.stack(xs).to(DEVICE)
    idx = t.randperm(real.shape[0], device=DEVICE)[:N_SAMPLES]
    real = real.index_select(0, idx)

    blurred = blur_images(real, BLUR_SIGMA)
    noise_low = make_subspace_noise(N_SAMPLES, LOW_DIM, NOISE_STD)
    noise_high = make_subspace_noise(N_SAMPLES, HIGH_DIM, NOISE_STD)
    noise_blur = blur_images(t.randn_like(real) * NOISE_STD, BLUR_SIGMA)

    spectra = {
        "MNIST": radial_power_spectrum(real),
        f"Blurred MNIST (σ={BLUR_SIGMA})": radial_power_spectrum(blurred),
        f"Low-dim noise (k={LOW_DIM})": radial_power_spectrum(noise_low),
        f"High-dim noise (k={HIGH_DIM})": radial_power_spectrum(noise_high),
        f"Blurred noise (σ={BLUR_SIGMA})": radial_power_spectrum(noise_blur),
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for label, spec in spectra.items():
        ax.plot(spec, label=label)
    ax.set_xlabel("Radial frequency bin")
    ax.set_ylabel("Power")
    ax.set_title("Radial Power Spectra")
    ax.set_yscale("log")
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()

    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    print(f"Current script name: {script_name}")
    fig.savefig(f"plots_a/{script_name}_line.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to plots_a/{script_name}_line.png")
