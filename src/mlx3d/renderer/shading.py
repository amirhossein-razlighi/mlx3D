"""Lighting, shaders, and a hard-rasterizer mesh renderer.

Mirrors the PyTorch3D ``Rasterizer + Shader`` split: :func:`rasterize_meshes`
resolves visibility, then a shader interpolates attributes and applies a
lighting model. :func:`render_mesh` ties them together into a single call that
satisfies the :class:`~mlx3d.renderer.Renderer` protocol and returns
``{"image", "alpha", "depth", "normals"}``.

Lighting is plain Lambert + Blinn-Phong in world space and is fully
differentiable, so light/material/vertex parameters can all be optimized.
An optional PBR path uses a compact Cook-Torrance/GGX shader for
metallic/roughness material previews.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from ..cameras import Camera
from ..structures import Meshes
from .mesh import sample_texture
from .protocols import RenderOutput
from .rasterizer import interpolate_face_attributes, rasterize_meshes

__all__ = [
    "AmbientLights",
    "DirectionalLights",
    "PointLights",
    "pbr_shading",
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


def _direct_light_terms(light: Light, points: mx.array) -> tuple[mx.array | None, mx.array]:
    if isinstance(light, AmbientLights):
        return None, light.ambient
    if isinstance(light, DirectionalLights):
        return light._to_light(points), _arr(light.color)
    return light._to_light(points), _arr(light.color)


def pbr_shading(
    points: mx.array,
    normals: mx.array,
    albedo: mx.array,
    camera_center: mx.array,
    lights: list[Light],
    roughness: float | mx.array = 0.5,
    metallic: float | mx.array = 0.0,
) -> mx.array:
    """Cook-Torrance/GGX material shading.

    This is a compact real-time PBR shader: GGX normal distribution, Schlick
    Fresnel, and Smith-Schlick geometry term. It is not a full environment-lit
    renderer, but it gives glTF-style base-color/metallic/roughness controls
    with stable MLX autodiff.
    """
    n = normals
    v = _arr(camera_center) - points
    v = v / mx.maximum(mx.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
    rough = mx.clip(_arr(roughness), 0.04, 1.0)
    metal = mx.clip(_arr(metallic), 0.0, 1.0)
    if rough.ndim == 0:
        rough = mx.broadcast_to(rough, albedo.shape[:-1] + (1,))
    if metal.ndim == 0:
        metal = mx.broadcast_to(metal, albedo.shape[:-1] + (1,))

    f0 = 0.04 * (1.0 - metal) + albedo * metal
    ambient = mx.zeros((3,))
    color = mx.zeros_like(albedo)
    alpha = rough * rough
    alpha2 = alpha * alpha
    k = ((rough + 1.0) ** 2) / 8.0
    n_dot_v = mx.maximum(mx.sum(n * v, axis=-1, keepdims=True), 1e-5)

    for light in lights:
        light_dir, light_color = _direct_light_terms(light, points)
        if light_dir is None:
            ambient = ambient + light_color
            continue
        h = light_dir + v
        h = h / mx.maximum(mx.linalg.norm(h, axis=-1, keepdims=True), 1e-8)
        n_dot_l = mx.maximum(mx.sum(n * light_dir, axis=-1, keepdims=True), 0.0)
        n_dot_h = mx.maximum(mx.sum(n * h, axis=-1, keepdims=True), 0.0)
        v_dot_h = mx.maximum(mx.sum(v * h, axis=-1, keepdims=True), 0.0)

        denom = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0
        D = alpha2 / mx.maximum(math.pi * denom * denom, 1e-8)
        Gv = n_dot_v / mx.maximum(n_dot_v * (1.0 - k) + k, 1e-8)
        Gl = n_dot_l / mx.maximum(n_dot_l * (1.0 - k) + k, 1e-8)
        F = f0 + (1.0 - f0) * ((1.0 - v_dot_h) ** 5)
        specular = (D * Gv * Gl * F) / mx.maximum(4.0 * n_dot_v * n_dot_l, 1e-6)
        diffuse = (1.0 - F) * (1.0 - metal) * albedo / math.pi
        color = color + (diffuse + specular) * light_color * n_dot_l

    color = color + ambient * albedo * (1.0 - metal)
    return mx.clip(color, 0.0, 1.0)


def render_mesh(
    camera: Camera,
    mesh_or_verts: Meshes | mx.array,
    faces: mx.array | None = None,
    verts_colors: mx.array | None = None,
    texture: mx.array | None = None,
    verts_uvs: mx.array | None = None,
    faces_uvs: mx.array | None = None,
    lights: list[Light] | None = None,
    shininess: float = 32.0,
    specular_strength: float = 0.3,
    roughness: float | mx.array = 0.5,
    metallic: float | mx.array = 0.0,
    background: tuple[float, float, float] | float = 0.0,
    shading: str = "phong",
    ssaa: int = 1,
) -> RenderOutput:
    """Render a mesh with the hard rasterizer and Blinn-Phong lighting.

    Albedo comes from a UV texture when ``texture`` is given, otherwise from
    ``verts_colors`` (default mid-grey). ``ssaa > 1`` supersamples (renders at
    ``ssaa x`` resolution and box-downsamples) for antialiased edges.

    Besides ``image``/``alpha``/``depth``/``normals`` the result includes render
    passes (AOVs): ``position`` (``(H, W, 3)`` world-space hit point) and
    ``face_id`` (``(H, W)`` nearest-face index, ``-1`` where empty).

    Args:
        camera: viewing camera.
        mesh_or_verts: a single-mesh :class:`~mlx3d.structures.Meshes` or ``(V, 3)``
            vertices (``faces`` then required).
        faces: ``(F, 3)`` indices when passing raw vertices.
        verts_colors: ``(V, 3)`` albedo; defaults to mid-grey.
        texture: ``(H, W, 3)`` diffuse texture; requires ``verts_uvs`` and ``faces_uvs``.
        verts_uvs: ``(VT, 2)`` UV coordinates.
        faces_uvs: ``(F, 3)`` per-corner indices into ``verts_uvs``.
        lights: light list; defaults to one key light + ambient. ``shading="none"``
            ignores lights and returns flat albedo.
        shading: ``"phong"``, ``"pbr"``, or ``"none"`` (unlit albedo).
        roughness: scalar or ``(H, W, 1)`` material roughness for ``shading="pbr"``.
        metallic: scalar or ``(H, W, 1)`` material metalness for ``shading="pbr"``.
        background: scalar or ``(3,)`` background color.

    Returns:
        ``{"image", "alpha", "depth", "normals"}``.
    """
    if ssaa > 1:
        big = _scale_camera(camera, ssaa)
        hi = render_mesh(
            big,
            mesh_or_verts,
            faces,
            verts_colors,
            texture,
            verts_uvs,
            faces_uvs,
            lights,
            shininess,
            specular_strength,
            roughness,
            metallic,
            background,
            shading,
            ssaa=1,
        )
        return _downsample_passes(hi, ssaa)

    mesh = mesh_or_verts if isinstance(mesh_or_verts, Meshes) else Meshes([mesh_or_verts], [faces])
    verts = mesh.verts_packed()

    frag = rasterize_meshes(camera, mesh)
    positions = interpolate_face_attributes(frag, verts)  # world-space AOV
    if texture is not None:
        if verts_uvs is None or faces_uvs is None:
            raise ValueError("verts_uvs and faces_uvs are required with a texture.")
        # Interpolate per-corner UVs over the fragments, then sample the texture.
        fidx = mx.where(frag.valid, frag.pix_to_face, 0)
        uv_tri = verts_uvs[faces_uvs.astype(mx.int32)[fidx]]  # (H, W, 3, 2)
        uv = mx.sum(frag.bary[..., None] * uv_tri, axis=-2)  # (H, W, 2)
        albedo = sample_texture(texture, uv) * frag.valid[..., None]
    else:
        if verts_colors is None:
            verts_colors = mx.full((verts.shape[0], 3), 0.7)
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
        if shading == "phong":
            image = phong_shading(
                positions,
                normals_px,
                albedo,
                camera.camera_center,
                lights,
                shininess,
                specular_strength,
            )
        elif shading == "pbr":
            image = pbr_shading(
                positions,
                normals_px,
                albedo,
                camera.camera_center,
                lights,
                roughness=roughness,
                metallic=metallic,
            )
        else:
            raise ValueError("shading must be 'phong', 'pbr', or 'none'.")

    alpha = frag.valid.astype(mx.float32)
    bg = _arr(background)
    if bg.ndim == 0:
        bg = mx.broadcast_to(bg, (3,))
    image = image * alpha[..., None] + bg * (1.0 - alpha[..., None])
    return {
        "image": image,
        "alpha": alpha,
        "depth": frag.zbuf,
        "normals": normals_px,
        "position": positions,
        "face_id": frag.pix_to_face,
    }


def _scale_camera(camera: Camera, s: int) -> Camera:
    """A copy of ``camera`` at ``s x`` resolution (intrinsics scaled to match)."""
    return Camera(
        R=camera.R,
        t=camera.t,
        fx=camera.fx * s,
        fy=camera.fy * s,
        cx=camera.cx * s,
        cy=camera.cy * s,
        width=camera.width * s,
        height=camera.height * s,
        znear=camera.znear,
        zfar=camera.zfar,
        orthographic=camera.orthographic,
        distortion=camera.distortion,
        fisheye=camera.fisheye,
    )


def _box_downsample(x: mx.array, s: int) -> mx.array:
    """Average non-overlapping ``s x s`` blocks of a ``(H, W, ...)`` array."""
    h, w = x.shape[:2]
    tail = x.shape[2:]
    x = x.reshape(h // s, s, w // s, s, *tail)
    return x.mean(axis=(1, 3))


def _downsample_passes(passes: RenderOutput, s: int) -> RenderOutput:
    """Box-downsample supersampled passes; face_id is nearest-sampled."""
    out: RenderOutput = {}
    for k, v in passes.items():
        if k == "face_id":
            out[k] = v[::s, ::s]  # discrete label: pick the block's top-left
        elif k == "normals":
            n = _box_downsample(v, s)
            out[k] = n / mx.maximum(mx.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
        else:
            out[k] = _box_downsample(v, s)
    return out
