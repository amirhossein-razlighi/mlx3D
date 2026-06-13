"""The renderer convention that ties MLX3D's pipeline together.

Every image-space renderer in MLX3D — :func:`render_mesh_soft`,
:func:`render_points`, :meth:`GaussianModel.render` — follows the same simple
contract: it is a callable that takes a :class:`~mlx3d.cameras.Camera` (plus
whatever scene data it needs) and returns a :class:`RenderOutput` dict.

This is intentionally a *convention*, not a base class. To add your own
renderer — a toon shader, a wireframe pass, a custom rasterizer — you just
write a function with the same shape and it drops straight into the rest of the
pipeline (saving, the viewer, side-by-side comparisons)::

    def render_normals(camera, mesh) -> RenderOutput:
        ...
        return {"image": rgb, "alpha": alpha, "depth": depth}

The :class:`Renderer` protocol exists only to document and type-check that
shape; you never need to inherit from anything.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

import mlx.core as mx

from ..cameras import Camera

__all__ = ["RenderOutput", "Renderer"]


class RenderOutput(TypedDict, total=False):
    """The dict every image-space renderer returns.

    Keys are optional so depth-only or alpha-only passes are still valid
    outputs, but RGB renderers populate all three:

    - ``image``: ``(H, W, 3)`` RGB in ``[0, 1]``.
    - ``alpha``: ``(H, W)`` coverage / opacity in ``[0, 1]``.
    - ``depth``: ``(H, W)`` expected depth in world units.

    Renderers may attach extra keys (e.g. ``means2d`` for splatting); consumers
    should read by key rather than assume an exact set.
    """

    image: mx.array
    alpha: mx.array
    depth: mx.array


@runtime_checkable
class Renderer(Protocol):
    """Anything callable as ``renderer(camera, *scene) -> RenderOutput``.

    Use it purely as a type hint for code that accepts a pluggable renderer::

        def turntable(renderer: Renderer, cameras, scene): ...

    Both library functions and your own closures satisfy it without subclassing.
    """

    def __call__(self, camera: Camera, *args: object, **kwargs: object) -> RenderOutput: ...
