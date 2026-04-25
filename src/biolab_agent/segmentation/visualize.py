"""Helpers for decoding RLE masks and rendering overlays for the UI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def decode_rle(rle: str, height: int, width: int) -> np.ndarray:
    """Inverse of ``segment.py::_encode_rle``  -  flat row-major binary RLE."""
    if not rle:
        return np.zeros((height, width), dtype=bool)
    runs = [int(x) for x in rle.split(",") if x]
    flat = np.zeros(height * width, dtype=bool)
    cur = False
    pos = 0
    for run in runs:
        if cur:
            flat[pos : pos + run] = True
        cur = not cur
        pos += run
    return flat[: height * width].reshape((height, width))


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 80, 0),
    alpha: float = 0.45,
) -> Image.Image:
    """Return a new RGB image with ``mask`` painted in ``color``."""
    img = image.convert("RGB")
    if mask.shape != (img.height, img.width):
        # Resample the mask to match the image (nearest neighbour keeps edges crisp).
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
            (img.width, img.height), resample=Image.Resampling.NEAREST
        )
        mask_np = np.asarray(mask_img) > 0
    else:
        mask_np = mask

    arr = np.asarray(img).astype(np.float32)
    overlay = np.zeros_like(arr)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color
    blended = np.where(
        mask_np[..., None],
        arr * (1 - alpha) + overlay * alpha,
        arr,
    ).astype(np.uint8)
    return Image.fromarray(blended)


def render_segmentation_overlay(
    image_path: Path,
    rle: str,
    height: int,
    width: int,
) -> Image.Image:
    img = Image.open(image_path)
    mask = decode_rle(rle, height, width)
    return overlay_mask(img, mask)
