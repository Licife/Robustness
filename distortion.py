"""
distortion.py

Multi-distortion simulation module for ABINN image steganography training/testing.

Supported distortions:
1. Poisson noise (variance/intensity level = 3)
2. Gaussian noise (sigma = 1, 5)
3. JPEG compression (quality factor = 50, 90)

Input:
    Tensor image:
        shape: [B, C, H, W]
        range: [0, 1]

Output:
    Tensor image:
        shape: [B, C, H, W]
        range: [0, 1]

Usage:
    from distortion import apply_distortion

    stego_dist = apply_distortion(
        stego,
        distortion_type="gaussian5"
    )

"""

import io
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ============================================================
# Basic operations
# ============================================================

def _clamp(img):
    """Keep image range in [0,1]."""
    return torch.clamp(img, 0.0, 1.0)


# ============================================================
# Gaussian Noise
# ============================================================

def gaussian_noise(img, sigma=1):
    """
    Additive Gaussian noise.

    sigma:
        Noise standard deviation under 0-255 image scale.

    Example:
        sigma=1
        sigma=5
    """

    noise = torch.randn_like(img) * (sigma / 255.0)

    return _clamp(img + noise)


# ============================================================
# Poisson Noise
# ============================================================

def poisson_noise(img, variance_level=3):
    """
    Poisson noise.

    variance_level:
        Controls noise intensity.

    Here:
        variance_level=3

    Image is converted to 0-255 scale,
    Poisson sampling is performed,
    then converted back.
    """

    img = torch.clamp(img,0,1)

    img_255 = img * 255.0

    scale = max(float(variance_level),1.0)

    noisy = torch.poisson(img_255 * scale) / scale

    return torch.clamp(noisy / 255.0,0,1)


# ============================================================
# JPEG Compression
# ============================================================

def jpeg_compression(img, quality=90):
    """
    JPEG compression simulation.

    quality:
        JPEG quality factor.

        50 -> stronger compression
        90 -> lighter compression
    """

    device = img.device

    img_cpu = img.detach().cpu()

    results = []

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    for b in range(img_cpu.shape[0]):

        pil_img = to_pil(img_cpu[b])

        buffer = io.BytesIO()

        pil_img.save(
            buffer,
            format="JPEG",
            quality=quality
        )

        buffer.seek(0)

        compressed = Image.open(buffer).convert("RGB")

        results.append(
            to_tensor(compressed)
        )

    out = torch.stack(results)

    return out.to(device)


# ============================================================
# Distortion Pool
# ============================================================

DISTORTION_POOL = [
    "poisson3",
    "gaussian1",
    "gaussian5",
    "jpeg50",
    "jpeg90"
]


def apply_distortion(
        img,
        distortion_type=None,
        random_select=False
):
    """
    Main interface.

    Parameters
    ----------
    img:
        Tensor [B,C,H,W], range [0,1]

    distortion_type:
        "poisson3"
        "gaussian1"
        "gaussian5"
        "jpeg50"
        "jpeg90"

    random_select:
        If True:
            randomly select one distortion
            from distortion pool.

    """

    if random_select or distortion_type is None:

        distortion_type = random.choice(
            DISTORTION_POOL
        )


    if distortion_type == "poisson3":

        return poisson_noise(
            img,
            variance_level=3
        )


    elif distortion_type == "gaussian1":

        return gaussian_noise(
            img,
            sigma=1
        )


    elif distortion_type == "gaussian5":

        return gaussian_noise(
            img,
            sigma=5
        )


    elif distortion_type == "jpeg50":

        return jpeg_compression(
            img,
            quality=50
        )


    elif distortion_type == "jpeg90":

        return jpeg_compression(
            img,
            quality=90
        )


    else:

        raise ValueError(
            f"Unknown distortion type: {distortion_type}"
        )



# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":

    x = torch.rand(
        2,3,256,256
    )

    y = apply_distortion(
        x,
        random_select=True
    )

    print(
        "Input:",
        x.shape,
        x.min().item(),
        x.max().item()
    )

    print(
        "Output:",
        y.shape,
        y.min().item(),
        y.max().item()
    )
