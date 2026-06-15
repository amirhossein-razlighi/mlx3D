"""Lighting, shaders, and a hard-rasterizer mesh renderer.

Mirrors the PyTorch3D ``Rasterizer + Shader`` split: :func:`rasterize_meshes`
resolves visibility, then a shader interpolates attributes and applies a
lighting model. :func:`render_mesh` ties them together into a single call that
satisfies the :class:`~mlx3d.renderer.Renderer` protocol and returns
``{"image", "alpha", "depth", "normals"}``.

Lighting is plain Lambert + Blinn-Phong in world space and is fully
differentiable, so light/material/vertex parameters can all be optimized.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from ..cameras import Camera
from ..structures import Meshes
from .protocols import RenderOutput
from .rasterizer import interpolate_face_attributes, rasterize_meshes

__all__ = [
    "AmbientLights",
    "DirectionalLights",
    "PointLights",
    "phong_shading",
    "render_mesh",
]


def _arr(x) -> mx.array:
    return mx.array(x, dtype=mx.float32)


@dataclass
class AmbientLights:
    """Uniform ambient illumination."""

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def diffuse(self, normals: mx.array, points: mx.array) -> mx.array:
        return mx.zeros_like(normals)

    def specular(self, normals, points, camera_center, shininess) -> mx.array:
        return mx.zeros_like(normals)

    @property
    def ambient(self) -> mx.array:
        return _arr(self.color)


@dataclass
class DirectionalLights:
    """A light at infinity. ``direction`` is the direction the light travels."""

    direction: tuple[float, float, float] = (0.0, -1.0, 0.0)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient_color: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def _to_light(self, points: mx.array) -> mx.array:
        d = -_arr(self.direction)
        d = d / mx.maximum(mx.linalg.norm(d), 1e-8)
        return mx.broadcast_to(d, points.shape)

    def diffuse(self, normals: mx.array, points: mx.array) -> mx.array:
        light_dir = self._to_light(points)
        ndl = mx.maximum(mx.sum(normals * light_dir, axis=-1, keepdims=True), 0.0)
        return ndl * _arr(self.color)

    def specular(self, normals, points, camera_center, shininess) -> mx.array:
        return _blinn_phong(
            normals, points, self._to_light(points), camera_center, shininess, _arr(self.color)
        )

    @property
    def ambient(self) -> mx.array:
        return _arr(self.ambient_color)


@dataclass
class PointLights:
    """A point light at ``location`` (world space)."""

    location: tuple[float, float, float] = (0.0, 1.0, -3.0)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient_color: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def _to_light(self, points: mx.array) -> mx.array:
        d = _arr(self.location) - points
        return d / mx.maximum(mx.linalg.norm(d, axis=-1, keepdims=True), 1e-8)

    def diffuse(self, normals: mx.array, points: mx.array) -> mx.array:
        ndl = mx.maximum(mx.sum(normals * self._to_light(points), axis=-1, keepdims=True), 0.0)
        return ndl * _arr(self.color)

    def specular(self, normals, points, camera_center, shininess) -> mx.array:
        return _blinn_phong(
            normals, points, self._to_light(points), camera_center, shininess, _arr(self.color)
        )

    @property
    def ambient(self) -> mx.array:
        return _arr(self.ambient_color)


Light = AmbientLights | DirectionalLights | PointLights


def _blinn_phong(normals, points, light_dir, camera_center, shininess, light_color) -> mx.array:
    view_dir = _arr(camera_center) - points
    view_dir = view_dir / mx.maximum(mx.linalg.norm(view_dir, axis=-1, keepdims=True), 1e-8)
    half = light_dir + view_dir
    half = half / mx.maximum(mx.linalg.norm(half, axis=-1, keepdims=True), 1e-8)
    ndh = mx.maximum(mx.sum(normals * half, axis=-1, keepdims=True), 0.0)
    # Only light front faces (n.l > 0).
    ndl = mx.sum(normals * light_dir, axis=-1, keepdims=True)
    return mx.power(ndh, shininess) * (ndl > 0) * light_color


def phong_shading(
    points: mx.array,
    normals: mx.array,
    albedo: mx.array,
    camera_center: mx.array,
    lights: list[Light],
    shininess: float = 32.0,
    specular_strength: float = 0.3,
) -> mx.array:
    """Blinn-Phong shade per-pixel buffers.

    Args:
        points: ``(H, W, 3)`` world positions.
        normals: ``(H, W, 3)`` unit world normals.
        albedo: ``(H, W, 3)`` base color.
        camera_center: ``(3,)`` camera position.
        lights: list of light sources.
        shininess: specular exponent.
        specular_strength: scalar weight on the specular term.
    """
    ambient = mx.zeros((3,))
    diffuse = mx.zeros_like(points)
    specular = mx.zeros_like(points)
    for light in lights:
        ambient = ambient + light.ambient
        diffuse = diffuse + light.diffuse(normals, points)
        specular = specular + light.specular(normals, points, camera_center, shininess)
    color = albedo * (ambient + diffuse) + specular_strength * specular
    return mx.clip(color, 0.0, 1.0)


def render_mesh(
    camera: Camera,
    mesh_or_verts: Meshes | mx.array,
    faces: mx.array | None = None,
    verts_colors: mx.array | None = None,
    lights: list[Light] | None = None,
    shininess: float = 32.0,
    specular_strength: float = 0.3,
    background: tuple[float, float, float] | float = 0.0,
    shading: str = "phong",
) -> RenderOutput:
    """Render a mesh with the hard rasterizer and Blinn-Phong lighting.

    Args:
        camera: viewing camera.
        mesh_or_verts: a single-mesh :class:`~mlx3d.structures.Meshes` or ``(V, 3)``
            vertices (``faces`` then required).
        faces: ``(F, 3)`` indices when passing raw vertices.
        verts_colors: ``(V, 3)`` albedo; defaults to mid-grey.
        lights: light list; defaults to one key light + ambient. ``shading="none"``
            ignores lights and returns flat albedo.
        shading: ``"phong"`` or ``"none"`` (unlit albedo).
        background: scalar or ``(3,)`` background color.

    Returns:
        ``{"image", "alpha", "depth", "normals"}``.
    """
    mesh = mesh_or_verts if isinstance(mesh_or_verts, Meshes) else Meshes([mesh_or_verts], [faces])
    verts = mesh.verts_packed()
    if verts_colors is None:
        verts_colors = mx.full((verts.shape[0], 3), 0.7)

    frag = rasterize_meshes(camera, mesh)
    albedo = interpolate_face_attributes(frag, verts_colors)

    if shading == "none":
        image = albedo
        normals_px = mx.zeros_like(albedo)
    else:
        if lights is None:
            lights = [
                DirectionalLights(direction=(-1.0, -1.0, -0.6), color=(1.0, 1.0, 1.0)),
                AmbientLights(color=(0.25, 0.25, 0.25)),
            ]
        positions = interpolate_face_attributes(frag, verts)
        vnormals = mesh.verts_normals_packed()
        normals_px = interpolate_face_attributes(frag, vnormals)
        normals_px = normals_px / mx.maximum(
            mx.linalg.norm(normals_px, axis=-1, keepdims=True), 1e-8
        )
        # Two-sided shading: orient each normal toward the camera so meshes with
        # inward/inconsistent winding still light correctly (a black mesh from a
        # flipped normal is the most common surprise otherwise).
        view_dir = _arr(camera.camera_center) - positions
        facing = mx.sum(normals_px * view_dir, axis=-1, keepdims=True) < 0
        normals_px = mx.where(facing, -normals_px, normals_px)
        image = phong_shading(
            positions,
            normals_px,
            albedo,
            camera.camera_center,
            lights,
            shininess,
            specular_strength,
        )

    alpha = frag.valid.astype(mx.float32)
    bg = _arr(background)
    if bg.ndim == 0:
        bg = mx.broadcast_to(bg, (3,))
    image = image * alpha[..., None] + bg * (1.0 - alpha[..., None])
    return {"image": image, "alpha": alpha, "depth": frag.zbuf, "normals": normals_px}
