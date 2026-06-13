"""Reading and writing images as MLX arrays.

These helpers bridge MLX float images (``(H, W, 3)`` or ``(H, W)`` in ``[0, 1]``)
and on-disk PNG/JPEG files, so any renderer output can be saved or loaded with a
single call::

    out = render_mesh_soft(camera, mesh)
    save_image("render.png", out["image"])
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
from PIL import Image

__all__ = ["load_image", "save_image"]


def load_image(path: str, *, normalize: bool = True) -> mx.array:
    """Load an image from disk as an MLX array.

    Args:
        path: Path to a PNG/JPEG/... file readable by Pillow.
        normalize: If ``True`` (default), return float32 in ``[0, 1]``;
            otherwise return the raw ``uint8`` values.

    Returns:
        ``(H, W, 3)`` for RGB images (alpha is dropped), or ``(H, W)`` for
        single-channel images.
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    arr = np.asarray(img)
    if not normalize:
        return mx.array(arr)
    return mx.array(arr.astype(np.float32) / 255.0)


def save_image(path: str, image: mx.array | np.ndarray) -> None:
    """Save an MLX (or NumPy) image to disk.

    Accepts float images in ``[0, 1]`` (clipped) or ``uint8`` images. Grayscale
    ``(H, W)``, RGB ``(H, W, 3)`` and RGBA ``(H, W, 4)`` are all supported.
    Parent directories are created automatically.

    Args:
        path: Destination file path; the format is inferred from the extension.
        image: The image to save.
    """
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    Image.fromarray(arr).save(path)
