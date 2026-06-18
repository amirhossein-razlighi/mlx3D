"""Render Gaussian checkpoints and mesh assets from the command line."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import mlx.core as mx

from ..cameras import Camera
from ..io import load_gltf, load_obj, load_ply, save_image
from ..io.gltf_io import GltfData
from ..renderer import render_mesh
from ..splatting import GaussianModel
from ..structures import Meshes
from ..transforms import quaternion_to_matrix


@dataclass
class _MeshAsset:
    mesh: Meshes
    colors: mx.array | None = None
    uvs: mx.array | None = None
    face_uvs: mx.array | None = None
    texture: mx.array | None = None
    roughness: float | None = None
    metallic: float | None = None


def _vec3(values: list[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{name} expects exactly 3 values.")
    return (float(values[0]), float(values[1]), float(values[2]))


def _camera(args: argparse.Namespace) -> Camera:
    return Camera.look_at(
        eye=_vec3(args.eye, "--eye"),
        at=_vec3(args.at, "--at"),
        up=_vec3(args.up, "--up"),
        fov=args.fov,
        width=args.width,
        height=args.height,
    )


def _depth_to_rgb(depth: mx.array, alpha: mx.array) -> mx.array:
    valid = alpha > 1e-6
    near = mx.min(mx.where(valid, depth, mx.full(depth.shape, 1e9, dtype=depth.dtype)))
    far = mx.max(mx.where(valid, depth, mx.zeros_like(depth)))
    denom = mx.maximum(far - near, 1e-6)
    shade = 1.0 - (depth - near) / denom
    return (
        mx.broadcast_to(mx.clip(shade, 0.0, 1.0)[..., None], (*depth.shape, 3)) * alpha[..., None]
    )


def _normal_to_rgb(normals: mx.array, alpha: mx.array) -> mx.array:
    return (normals * 0.5 + 0.5) * alpha[..., None]


def _is_gaussian_ply(path: str) -> bool:
    if not path.lower().endswith(".ply"):
        return False
    try:
        data = load_ply(path)
    except Exception:
        return False
    extra = data.extra
    needed = {"f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2"}
    return needed.issubset(extra.keys())


def _uniform_gltf_pbr_factors(data: GltfData) -> tuple[float | None, float | None]:
    if not data.materials:
        return None, None
    if data.material_ids is None:
        material_ids = {0} if len(data.materials) == 1 else set()
    else:
        material_ids = {int(i) for i in data.material_ids.tolist()}
    material_ids = {i for i in material_ids if 0 <= i < len(data.materials)}
    if not material_ids:
        return None, None
    roughness = {data.materials[i].roughness_factor for i in material_ids}
    metallic = {data.materials[i].metallic_factor for i in material_ids}
    return (
        roughness.pop() if len(roughness) == 1 else None,
        metallic.pop() if len(metallic) == 1 else None,
    )


def _load_mesh(path: str) -> _MeshAsset:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".glb", ".gltf"}:
        data = load_gltf(path)
        mesh = Meshes([data.verts], [data.faces])
        verts_colors = None
        if data.texture_image is None and data.material_ids is not None and data.materials:
            colors = mx.zeros((data.faces.shape[0], 3), dtype=mx.float32)
            for mat_id, mat in enumerate(data.materials):
                face_mask = data.material_ids == mat_id
                color = mx.array(mat.base_color[:3], dtype=mx.float32)
                colors = mx.where(face_mask[:, None], color, colors)
            verts_colors = mx.zeros((data.verts.shape[0], 3), dtype=mx.float32)
            counts = mx.zeros((data.verts.shape[0], 1), dtype=mx.float32)
            for k in range(3):
                verts_colors = verts_colors.at[data.faces[:, k]].add(colors)
                counts = counts.at[data.faces[:, k]].add(mx.ones((data.faces.shape[0], 1)))
            verts_colors = verts_colors / mx.maximum(counts, 1.0)
        roughness, metallic = _uniform_gltf_pbr_factors(data)
        return _MeshAsset(
            mesh,
            colors=verts_colors,
            uvs=data.uvs,
            face_uvs=data.faces,
            texture=data.texture_image,
            roughness=roughness,
            metallic=metallic,
        )
    if ext == ".obj":
        data = load_obj(path)
        mesh = Meshes([data.verts], [data.faces])
        return _MeshAsset(
            mesh,
            colors=data.verts_colors,
            uvs=data.texcoords,
            face_uvs=data.faces_texcoords_idx,
            texture=data.texture_image,
        )
    if ext == ".ply":
        data = load_ply(path)
        if data.faces is None:
            raise ValueError("PLY mesh rendering requires face indices.")
        return _MeshAsset(Meshes([data.verts], [data.faces]), colors=data.colors)
    raise ValueError(f"Unsupported mesh input extension: {ext}")


def _render_gaussian(args: argparse.Namespace) -> mx.array:
    model = GaussianModel.load_ply(args.input)
    cam = _camera(args)
    bg = mx.array(args.background, dtype=mx.float32)
    if args.mode == "depth":
        out = model.render_depth(cam, antialias=args.antialias, projection=args.projection)
        return _depth_to_rgb(out["depth"], out["alpha"])
    out = model.render(cam, background=bg, antialias=args.antialias, projection=args.projection)
    if args.mode == "normal":
        normals = quaternion_to_matrix(model.params["quats"])[:, :, 2]
        normals = model.render_features(
            cam,
            normals,
            normalize=True,
            antialias=args.antialias,
            projection=args.projection,
        )
        return _normal_to_rgb(normals["features"], normals["alpha"])
    return out["image"]


def _render_mesh(args: argparse.Namespace) -> mx.array:
    asset = _load_mesh(args.input)
    cam = _camera(args)
    shading = args.shading if args.shading is not None else ("none" if args.unlit else "phong")
    material_kwargs = {}
    if asset.roughness is not None:
        material_kwargs["roughness"] = asset.roughness
    if asset.metallic is not None:
        material_kwargs["metallic"] = asset.metallic
    out = render_mesh(
        cam,
        asset.mesh,
        verts_colors=asset.colors,
        texture=asset.texture,
        verts_uvs=asset.uvs,
        faces_uvs=asset.face_uvs,
        background=tuple(args.background),
        shading=shading,
        ssaa=args.ssaa,
        **material_kwargs,
    )
    if args.mode == "depth":
        return _depth_to_rgb(out["depth"], out["alpha"])
    if args.mode == "normal":
        return _normal_to_rgb(out["normals"], out["alpha"])
    return out["image"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Gaussian PLY checkpoint or mesh asset")
    parser.add_argument("--out", required=True, help="output PNG/JPEG path")
    parser.add_argument("--type", choices=["auto", "gaussian", "mesh"], default="auto")
    parser.add_argument("--mode", choices=["rgb", "depth", "normal"], default="rgb")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--eye", type=float, nargs=3, default=(0.0, 0.0, -4.0))
    parser.add_argument("--at", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--up", type=float, nargs=3, default=(0.0, 1.0, 0.0))
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--antialias", action="store_true", help="Gaussian opacity compensation")
    parser.add_argument(
        "--projection",
        choices=["ewa", "ut"],
        default="ewa",
        help="Gaussian projection: fast EWA or 3DGUT-style Unscented Transform",
    )
    parser.add_argument("--ssaa", type=int, default=1, help="mesh supersampling factor")
    parser.add_argument("--shading", choices=["phong", "pbr", "none"], default=None)
    parser.add_argument("--unlit", action="store_true", help="render mesh albedo without lighting")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive.")
    if args.ssaa <= 0:
        parser.error("--ssaa must be positive.")
    if args.unlit and args.shading is not None:
        parser.error("--unlit cannot be combined with --shading.")

    kind = args.type
    if kind == "auto":
        kind = "gaussian" if _is_gaussian_ply(args.input) else "mesh"
    image = _render_gaussian(args) if kind == "gaussian" else _render_mesh(args)
    save_image(args.out, image)


if __name__ == "__main__":
    main()
