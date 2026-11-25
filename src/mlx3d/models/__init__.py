"""Pretrained 3D foundation models, ported to MLX.

Each model loads official weights (downloaded once from the Hugging Face Hub
with the ``models`` extra: ``pip install "mlx3d[models]"``) and returns
mlx3d structures — :class:`~mlx3d.cameras.Camera`,
:class:`~mlx3d.structures.Pointclouds` — so its output plugs straight into
rendering, training and capture.
"""

from .depth_anything3 import DepthAnything3, GeometryPrediction

__all__ = ["DepthAnything3", "GeometryPrediction"]
