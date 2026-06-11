"""MLX3D: differentiable 3D computer vision on Apple Silicon, built on MLX."""

__version__ = "0.1.0"

from . import (
    cameras,
    datasets,
    io,
    losses,
    nn,
    ops,
    renderer,
    splatting,
    structures,
    transforms,
    utils,
)

__all__ = [
    "__version__",
    "cameras",
    "datasets",
    "io",
    "losses",
    "nn",
    "ops",
    "renderer",
    "splatting",
    "structures",
    "transforms",
    "utils",
]
