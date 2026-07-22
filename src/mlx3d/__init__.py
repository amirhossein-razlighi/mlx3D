"""MLX3D: differentiable 3D computer vision on Apple Silicon, built on MLX."""

# The version is derived from the git tag at build time by hatch-vcs. Prefer
# the installed package metadata (authoritative for an installed wheel); fall
# back to the file hatch-vcs writes into the source tree, then to a sentinel
# for a bare source checkout that was never built or installed.
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("mlx3d")
except PackageNotFoundError:  # pragma: no cover - not installed
    try:
        from ._version import __version__
    except ImportError:
        __version__ = "0.0.0+unknown"

from . import (
    cameras,
    capture,
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
    viewer,
)

__all__ = [
    "__version__",
    "cameras",
    "capture",
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
    "viewer",
]
