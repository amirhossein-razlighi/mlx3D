"""Wavefront OBJ loading and saving.

``load_obj`` parses vertices, faces (polygons are fan-triangulated), optional
per-vertex normals/texture coordinates, and the non-standard but common
vertex-color extension (``v x y z r g b``).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

__all__ = ["load_obj", "save_obj", "ObjData"]


@dataclass
class ObjData:
    """Result of :func:`load_obj`.

    Attributes:
        verts: (V, 3) vertex positions.
        faces: (F, 3) triangle vertex indices.
        normals: (VN, 3) normal vectors from ``vn`` lines, or ``None``.
        texcoords: (VT, 2) texture coordinates from ``vt`` lines, or ``None``.
        faces_normals_idx: (F, 3) per-corner indices into ``normals``, or ``None``.
        faces_texcoords_idx: (F, 3) per-corner indices into ``texcoords``, or ``None``.
        verts_colors: (V, 3) vertex colors if the file used the color extension.
        texture_path: resolved diffuse texture path from ``.mtl`` / ``map_Kd``, or ``None``.
        texture_image: (H, W, 3) diffuse texture in [0, 1] when loaded, or ``None``.
    """

    verts: mx.array
    faces: mx.array
    normals: mx.array | None = None
    texcoords: mx.array | None = None
    faces_normals_idx: mx.array | None = None
    faces_texcoords_idx: mx.array | None = None
    verts_colors: mx.array | None = None
    texture_path: str | None = None
    texture_image: mx.array | None = None


def _resolve_index(idx: int, count: int) -> int:
    # OBJ indices are 1-based; negative indices count from the end.
    return idx - 1 if idx > 0 else count + idx


def _load_mtl_texture(mtl_path: Path) -> str | None:
    if not mtl_path.exists():
        return None
    with open(mtl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0] == "map_Kd":
                return str((mtl_path.parent / parts[1]).resolve())
    return None


def _read_texture(path: str) -> mx.array:
    img = Image.open(path).convert("RGB")
    return mx.array(np.asarray(img, dtype=np.float32) / 255.0)


def load_obj(path: str, load_texture: bool = True) -> ObjData:
    """Load a Wavefront OBJ file. Polygon faces are fan-triangulated."""
    path_obj = Path(path)
    verts: list[list[float]] = []
    colors: list[list[float]] = []
    normals: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    faces_vt: list[list[int]] = []
    faces_vn: list[list[int]] = []
    any_vt = False
    any_vn = False
    texture_path: str | None = None

    with open(path_obj, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "mtllib" and len(parts) > 1:
                texture_path = _load_mtl_texture(path_obj.parent / parts[1])
            elif tag == "v":
                vals = [float(x) for x in parts[1:]]
                verts.append(vals[:3])
                if len(vals) >= 6:
                    colors.append(vals[3:6])
            elif tag == "vn":
                normals.append([float(x) for x in parts[1:4]])
            elif tag == "vt":
                texcoords.append([float(x) for x in parts[1:3]])
            elif tag == "f":
                corner_v, corner_vt, corner_vn = [], [], []
                for corner in parts[1:]:
                    fields = corner.split("/")
                    corner_v.append(_resolve_index(int(fields[0]), len(verts)))
                    if len(fields) > 1 and fields[1]:
                        corner_vt.append(_resolve_index(int(fields[1]), len(texcoords)))
                        any_vt = True
                    else:
                        corner_vt.append(-1)
                    if len(fields) > 2 and fields[2]:
                        corner_vn.append(_resolve_index(int(fields[2]), len(normals)))
                        any_vn = True
                    else:
                        corner_vn.append(-1)
                # Fan triangulation for polygons with > 3 corners.
                for i in range(1, len(corner_v) - 1):
                    faces.append([corner_v[0], corner_v[i], corner_v[i + 1]])
                    faces_vt.append([corner_vt[0], corner_vt[i], corner_vt[i + 1]])
                    faces_vn.append([corner_vn[0], corner_vn[i], corner_vn[i + 1]])

    if not verts:
        raise ValueError(f"No vertices found in {path!r}.")

    return ObjData(
        verts=mx.array(np.asarray(verts, dtype=np.float32)),
        faces=mx.array(np.asarray(faces, dtype=np.int32).reshape(-1, 3)),
        normals=mx.array(np.asarray(normals, dtype=np.float32)) if normals else None,
        texcoords=mx.array(np.asarray(texcoords, dtype=np.float32)) if texcoords else None,
        faces_normals_idx=(
            mx.array(np.asarray(faces_vn, dtype=np.int32).reshape(-1, 3)) if any_vn else None
        ),
        faces_texcoords_idx=(
            mx.array(np.asarray(faces_vt, dtype=np.int32).reshape(-1, 3)) if any_vt else None
        ),
        verts_colors=(
            mx.array(np.asarray(colors, dtype=np.float32))
            if len(colors) == len(verts) and colors
            else None
        ),
        texture_path=texture_path,
        texture_image=_read_texture(texture_path) if texture_path and load_texture else None,
    )


def save_obj(
    path: str,
    verts: mx.array,
    faces: mx.array,
    verts_colors: mx.array | None = None,
) -> None:
    """Save a triangle mesh as a Wavefront OBJ file."""
    v = np.array(verts, dtype=np.float64)
    f = np.array(faces, dtype=np.int64)
    c = np.array(verts_colors, dtype=np.float64) if verts_colors is not None else None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as out:
        for i, p in enumerate(v):
            if c is not None:
                out.write(
                    f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {c[i][0]:.6f} {c[i][1]:.6f} {c[i][2]:.6f}\n"
                )
            else:
                out.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        for tri in f:
            out.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")
