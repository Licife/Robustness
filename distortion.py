"""RGB attacks for tensors [B, 3, H, W] on the normalized [0, 1] scale.

Gaussian sigma is measured on the 0-255 scale. Poisson uses peak=765
(the original scale=3), not a constant variance of 3.
The interface always returns one tensor. Set differentiable=True during
training to use an identity straight-through estimator for JPEG/Poisson.
"""

import io
import random

import torch
from PIL import Image
from torchvision import transforms


DISTORTION_POOL = ["poisson", "gaussian1", "gaussian10", "jpeg40", "jpeg90"]


def gaussian_noise(img, sigma=1):
    return (img.clamp(0, 1) + torch.randn_like(img) * (sigma / 255.0)).clamp(0, 1)


def poisson_noise(img, variance_level=3):
    """Legacy argument name; larger scale means weaker noise.

    Before output clipping, Var(y|x) = x / (255 * scale).
    """
    scale = max(float(variance_level), 1.0)
    x = img.clamp(0, 1)
    return (torch.poisson(x * (255.0 * scale)) / (255.0 * scale)).clamp(0, 1)


def jpeg_compression(img, quality=90):
    """Real PIL JPEG round trip; no intrinsic gradient through the codec."""
    img_cpu = img.detach().clamp(0, 1).float().cpu()
    to_pil, to_tensor = transforms.ToPILImage(), transforms.ToTensor()
    results = []
    for sample in img_cpu:
        with io.BytesIO() as buffer:
            to_pil(sample).save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                results.append(to_tensor(decoded.convert("RGB")))
    return torch.stack(results).to(device=img.device, dtype=img.dtype)


def apply_distortion(img, distortion_type=None, random_select=False, *, differentiable=False):
    """Apply one attack to a whole batch, returning a tensor of the same shape.

    STE preserves the attacked forward values but approximates the attack
    derivative as identity. Input clamping retains its normal derivative.
    Gaussian retains its ordinary additive-noise/clipping gradient.
    Legacy names remain available explicitly, but are not sampled in training.
    """
    if random_select or distortion_type is None:
        distortion_type = random.choice(DISTORTION_POOL)
    x = img.clamp(0, 1)
    if distortion_type == "none":
        return x
    if distortion_type in ("gaussian1", "gaussian5", "gaussian10"):
        return gaussian_noise(x, sigma=int(distortion_type[len("gaussian"):]))
    if distortion_type in ("poisson", "poisson3"):
        with torch.no_grad():
            attacked = poisson_noise(x, variance_level=3)
    elif distortion_type in ("jpeg40", "jpeg50", "jpeg90"):
        attacked = jpeg_compression(x, quality=int(distortion_type[len("jpeg"):]))
    else:
        raise ValueError(f"Unknown distortion type: {distortion_type}")
    if differentiable and torch.is_grad_enabled():
        return attacked + (x - x.detach())
    return attacked
