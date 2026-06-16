"""Image storage policies for dataset loaders.

Training views dominate the memory footprint of view-synthesis training:
250 full-resolution photos held as float32 are ~6 GB before a single
Gaussian exists. :class:`ImageCollection` behaves like a list of
``(H, W, 3)`` float32 MLX arrays but can store the underlying data three
ways:

- ``"ram"``: decoded float32 MLX arrays (fastest, 12 bytes/pixel).
- ``"uint8"``: decoded uint8 NumPy arrays, converted to float32 per access
  (4x smaller, conversion costs well under a millisecond per view).
- ``"disk"``: only file paths; images are decoded on every access
  (near-zero resident memory, ~10-30 ms JPEG/PNG decode per view).
"""

import mlx.core as mx
import numpy as np

__all__ = ["ImageCollection", "decode_image_file"]


def decode_image_file(
    path: str,
    downscale: int = 1,
    white_background: bool | None = None,
) -> np.ndarray:
    """Decode an image file to (H, W, 3) float32 in [0, 1].

    ``white_background`` controls RGBA compositing: ``None`` forces RGB,
    otherwise the alpha channel is composited onto white (True) or black
    (False), as needed for the Blender-synthetic renders.
    """
    from PIL import Image

    img = Image.open(path)
    if white_background is None:
        img = img.convert("RGB")
    if downscale > 1:
        img = img.resize((img.width // downscale, img.height // downscale), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 3 and arr.shape[-1] == 4:
        rgb, a = arr[..., :3], arr[..., 3:]
        bg = 1.0 if white_background else 0.0
        arr = rgb * a + bg * (1.0 - a)
    return arr[..., :3]


class ImageCollection:
    """A list-like view over training images with configurable storage.

    Always yields (H, W, 3) float32 MLX arrays in [0, 1], regardless of the
    underlying storage mode.
    """

    def __init__(
        self, cache: str = "ram", downscale: int = 1, white_background: bool | None = None
    ):
        if cache not in ("ram", "uint8", "disk"):
            raise ValueError(f"cache must be 'ram', 'uint8' or 'disk', got {cache!r}.")
        self.cache = cache
        self.downscale = downscale
        self.white_background = white_background
        self._items: list = []

    def append_file(self, path: str) -> None:
        """Register an image file, decoding it now or later per the cache mode."""
        if self.cache == "disk":
            self._items.append(path)
        else:
            arr = decode_image_file(path, self.downscale, self.white_background)
            if self.cache == "uint8":
                self._items.append((arr * 255.0 + 0.5).astype(np.uint8))
            else:
                self._items.append(mx.array(arr))

    def shape_of(self, i: int) -> tuple[int, int]:
        """(height, width) of image ``i`` (decodes it in disk mode)."""
        img = self[i]
        return img.shape[0], img.shape[1]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> mx.array:
        item = self._items[i]
        if self.cache == "ram":
            return item
        if self.cache == "uint8":
            return mx.array(item.astype(np.float32) / 255.0)
        return mx.array(decode_image_file(item, self.downscale, self.white_background))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def nbytes_resident(self) -> int:
        """Approximate resident memory of the stored images, in bytes."""
        if self.cache == "disk":
            return 0
        if self.cache == "uint8":
            return sum(a.nbytes for a in self._items)
        return sum(a.size * 4 for a in self._items)
