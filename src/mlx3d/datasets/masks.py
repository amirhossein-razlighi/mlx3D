"""Per-image foreground masks, aligned with an :class:`ImageCollection`.

Masks follow COLMAP's layout, so one mask set can serve training, evaluation
and feature extraction::

    scene/images/abc/frame_001.jpg
    scene/masks/abc/frame_001.jpg.png      (COLMAP convention)
    scene/masks/abc/frame_001.png          (also accepted)

A mask file may be 8-bit grayscale, RGB (luminance is used) or RGBA (alpha is
used). Alternatively the alpha channel of RGBA source images can be used
directly. Values are floats in ``[0, 1]``: 0 = background, 1 = foreground,
intermediate values are soft weights / partial opacity.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

__all__ = ["MaskCollection", "decode_mask_file", "find_mask_file"]


def find_mask_file(masks_dir: str, image_name: str) -> str | None:
    """Mask path for ``image_name`` (relative to the images directory), or None."""
    stem = os.path.splitext(image_name)[0]
    for candidate in (f"{image_name}.png", f"{stem}.png"):
        path = os.path.join(masks_dir, candidate)
        if os.path.isfile(path):
            return path
    return None


def decode_mask_file(path: str, downscale: int = 1, from_image_alpha: bool = False) -> np.ndarray:
    """Decode a mask to ``(H, W)`` float32 in ``[0, 1]``.

    With ``from_image_alpha`` the file is a source image and its alpha
    channel is the mask (it must have one).
    """
    from PIL import Image

    with Image.open(path) as img:
        has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
        if from_image_alpha and not has_alpha:
            raise ValueError(f"{path} has no alpha channel to use as a mask.")
        mask = img.convert("RGBA").getchannel("A") if has_alpha else img.convert("L")
        if downscale > 1:
            size = (mask.width // downscale, mask.height // downscale)
            mask = mask.resize(size, Image.Resampling.BOX)
        return np.asarray(mask, dtype=np.float32) / 255.0


class MaskCollection:
    """List-like ``(H, W)`` float32 masks with the same cache modes as images.

    ``"ram"`` keeps float32 MLX arrays, ``"uint8"`` keeps 8-bit NumPy arrays
    (masks are 8-bit at the source, so this is lossless), ``"disk"`` keeps
    paths and decodes on access.
    """

    def __init__(self, cache: str = "uint8", downscale: int = 1, from_image_alpha: bool = False):
        if cache not in ("ram", "uint8", "disk"):
            raise ValueError(f"cache must be 'ram', 'uint8' or 'disk', got {cache!r}.")
        self.cache = cache
        self.downscale = downscale
        self.from_image_alpha = from_image_alpha
        self._items: list = []

    def append_file(self, path: str, expected_hw: tuple[int, int] | None = None) -> None:
        """Register a mask, validating it now: it must be non-empty and, when
        ``expected_hw`` is given, match the image's ``(height, width)``."""
        arr = decode_mask_file(path, self.downscale, self.from_image_alpha)
        if expected_hw is not None and arr.shape != tuple(expected_hw):
            raise ValueError(
                f"Mask {path} is {arr.shape[1]}x{arr.shape[0]} but its image is "
                f"{expected_hw[1]}x{expected_hw[0]} (after downscale {self.downscale})."
            )
        if not arr.any():
            raise ValueError(f"Mask {path} is empty (all background).")
        if self.cache == "disk":
            self._items.append(path)
        elif self.cache == "uint8":
            self._items.append((arr * 255.0 + 0.5).astype(np.uint8))
        else:
            self._items.append(mx.array(arr))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> mx.array:
        item = self._items[i]
        if self.cache == "ram":
            return item
        if self.cache == "uint8":
            return mx.array(item.astype(np.float32) / 255.0)
        return mx.array(decode_mask_file(item, self.downscale, self.from_image_alpha))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def coverage(self, i: int) -> float:
        """Foreground fraction (mean mask value) of mask ``i``."""
        return float(self[i].mean())
