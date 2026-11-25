"""Image loading and resizing into DA3's input format."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import mlx.core as mx
import numpy as np
from PIL import Image

__all__ = ["ImageInput", "load_views", "processing_size"]

ImageInput = str | PathLike | Image.Image | np.ndarray

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _nearest_multiple(x: int, p: int) -> int:
    down = (x // p) * p
    return down + p if (down + p) - x <= x - down else max(down, p)


def processing_size(width: int, height: int, resolution: int, patch: int = 14) -> tuple[int, int]:
    """``(width, height)`` DA3 runs at: longest side scaled to ``resolution``,
    then each side rounded to the nearest multiple of ``patch``."""
    scale = resolution / max(width, height)
    w = max(1, round(width * scale))
    h = max(1, round(height * scale))
    return _nearest_multiple(w, patch), _nearest_multiple(h, patch)


def _to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    with Image.open(image) as im:
        return im.convert("RGB")


def load_views(
    images: Sequence[ImageInput], resolution: int = 504, patch: int = 14
) -> tuple[mx.array, np.ndarray, list[tuple[int, int]]]:
    """Load views, resize them to a shared processing size, and normalize.

    All views are resized to the size derived from the first one, so a
    sequence must share an aspect ratio (true for video frames and most
    captures).

    Returns:
        ``(pixels, rgb, original_sizes)``: ImageNet-normalized ``(1, S, H, W, 3)``
        float32 model input, the resized ``(S, H, W, 3)`` uint8 RGB frames, and
        each view's original ``(width, height)``.
    """
    if not images:
        raise ValueError("Need at least one image.")
    pil = [_to_pil(im) for im in images]
    sizes = [im.size for im in pil]
    w, h = processing_size(*sizes[0], resolution, patch)

    frames = []
    for im in pil:
        downscale = w <= im.width and h <= im.height
        resample = Image.Resampling.BOX if downscale else Image.Resampling.BICUBIC
        frames.append(np.asarray(im.resize((w, h), resample), dtype=np.uint8))
    rgb = np.stack(frames)
    pixels = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    return mx.array(pixels)[None], rgb, sizes
