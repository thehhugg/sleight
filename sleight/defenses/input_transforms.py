"""Input transformation defenses for adversarial robustness.

These defenses apply simple image transformations to inputs before
classification, which can remove or reduce adversarial perturbations.
They are model-agnostic and require no retraining.

Reference:
    Xu, W., Evans, D., & Qi, Y. (2018). "Feature Squeezing: Detecting
    Adversarial Examples in Deep Neural Networks." NDSS 2018.
    https://arxiv.org/abs/1704.01155
"""

from __future__ import annotations

import io

import numpy as np


def jpeg_compression(images: np.ndarray, quality: int = 75) -> np.ndarray:
    """Apply JPEG compression to images as a defense.

    JPEG compression removes high-frequency components that adversarial
    perturbations often rely on.

    Args:
        images: Input images as a NumPy array with shape (batch, H, W, C)
            and values in [0, 1].
        quality: JPEG quality factor (1-100). Lower values apply more
            compression and remove more perturbation, but also degrade
            image quality.

    Returns:
        Compressed images as a NumPy array with the same shape, values
        in [0, 1].
    """
    from PIL import Image

    result = np.zeros_like(images)
    for i in range(len(images)):
        img = images[i]
        # Handle grayscale (H, W, 1) and RGB (H, W, 3)
        if img.shape[-1] == 1:
            pil_img = Image.fromarray((img[:, :, 0] * 255).astype(np.uint8))
        else:
            pil_img = Image.fromarray((img * 255).astype(np.uint8))

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        pil_img = Image.open(buf)

        compressed = np.array(pil_img).astype(np.float32) / 255.0
        if img.shape[-1] == 1:
            compressed = compressed[..., np.newaxis]
        result[i] = compressed
    return result


def spatial_smoothing(images: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply spatial smoothing (mean filter) to images as a defense.

    Smoothing blurs the image, which can reduce the effect of small
    adversarial perturbations at the cost of some image detail.

    Args:
        images: Input images as a NumPy array with shape (batch, H, W, C)
            and values in [0, 1].
        kernel_size: Size of the smoothing kernel. Must be odd.

    Returns:
        Smoothed images as a NumPy array with the same shape, values
        in [0, 1].
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    pad = kernel_size // 2
    result = np.zeros_like(images)
    for i in range(len(images)):
        for c in range(images.shape[-1]):
            channel = images[i, :, :, c]
            padded = np.pad(channel, pad, mode="reflect")
            smoothed = np.zeros_like(channel)
            for dy in range(kernel_size):
                for dx in range(kernel_size):
                    smoothed += padded[dy : dy + channel.shape[0], dx : dx + channel.shape[1]]
            smoothed /= kernel_size * kernel_size
            result[i, :, :, c] = smoothed
    return np.clip(result, 0, 1).astype(np.float32)


def bit_depth_reduction(images: np.ndarray, bits: int = 4) -> np.ndarray:
    """Reduce the bit depth of images as a defense.

    Quantizes pixel values to fewer bits, which can remove fine-grained
    adversarial perturbations.

    Args:
        images: Input images as a NumPy array with shape (batch, H, W, C)
            and values in [0, 1].
        bits: Number of bits to retain (1-8). Lower values apply more
            aggressive quantization.

    Returns:
        Quantized images as a NumPy array with the same shape, values
        in [0, 1].
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be between 1 and 8, got {bits}")

    levels = 2**bits - 1
    quantized = np.round(images * levels) / levels
    return np.clip(quantized, 0, 1).astype(np.float32)
