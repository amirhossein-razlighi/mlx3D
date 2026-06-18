"""glTF 2.0 mesh IO (binary ``.glb`` and JSON ``.gltf``).

Loads triangle primitives from the default scene (positions, indices, optional
normals / UVs, node transforms, and material ids); saves a self-contained
``.glb``. glTF is the standard interchange format for real-world assets and
web viewers, so this unlocks loading and exporting meshes that tools actually
produce.

Coordinates are passed through unchanged (glTF is right-handed, +Y up).
"""

from __future__ import annotations

import base64
import io
import json
import os
import struct
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from PIL import Image

__all__ = ["GltfData", "GltfMaterial", "load_gltf", "save_gltf"]

# glTF accessor componentType / type codes.
_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_GLB_MAGIC = 0x46546C67  # "glTF"


@dataclass
class GltfMaterial:
    """A small material summary from glTF PBR metadata."""

    name: str | None = None
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    base_color_texture: int | None = None
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0


@dataclass
class GltfData:
    """Result of :func:`load_gltf`.

    Attributes:
        verts: ``(V, 3)`` positions.
        faces: ``(F, 3)`` triangle indices.
        normals: ``(V, 3)`` vertex normals, or ``None``.
        uvs: ``(V, 2)`` texture coordinates, or ``None``.
        material_ids: ``(F,)`` material index per face, or ``None``.
        materials: decoded material summaries.
        texture_image: first base-color texture image as ``(H, W, 3)`` in
            ``[0, 1]`` when the scene uses one; ``None`` otherwise.
    """

    verts: mx.array
    faces: mx.array
    normals: mx.array | None = None
    uvs: mx.array | None = None
    material_ids: mx.array | None = None
    materials: list[GltfMaterial] | None = None
    texture_image: mx.array | None = None


def _read_glb(path: str) -> tuple[dict, bytes]:
    with open(path, "rb") as f:
        data = f.read()
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError("Not a binary glTF (.glb) file.")
    if version != 2:
        raise ValueError(f"Unsupported glTF version {version}.")
    offset = 12
    gltf_json: dict | None = None
    bin_chunk = b""
    while offset < len(data):
        clen, ctype = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + clen]
        offset += clen
        if ctype == 0x4E4F534A:  # "JSON"
            gltf_json = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:  # "BIN\0"
            bin_chunk = chunk
    if gltf_json is None:
        raise ValueError("GLB has no JSON chunk.")
    return gltf_json, bin_chunk


def _buffer_bytes(gltf: dict, root_dir: str, glb_bin: bytes) -> list[bytes]:
    buffers = []
    for buf in gltf.get("buffers", []):
        uri = buf.get("uri")
        if uri is None:  # GLB binary chunk
            buffers.append(glb_bin)
        elif uri.startswith("data:"):
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            with open(os.path.join(root_dir, uri), "rb") as f:
                buffers.append(f.read())
    return buffers


def _read_accessor(gltf: dict, buffers: list[bytes], idx: int) -> np.ndarray:
    acc = gltf["accessors"][idx]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = _COMPONENT_DTYPE[acc["componentType"]]
    ncomp = _TYPE_NCOMP[acc["type"]]
    count = acc["count"]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    raw = buffers[view["buffer"]]
    stride = view.get("byteStride")
    if stride is None:
        arr = np.frombuffer(raw, dtype=dtype, count=count * ncomp, offset=start)
    else:
        item_bytes = np.dtype(dtype).itemsize * ncomp
        arr = np.empty((count, ncomp), dtype=dtype)
        for i in range(count):
            item = raw[start + i * stride : start + i * stride + item_bytes]
            arr[i] = np.frombuffer(item, dtype=dtype, count=ncomp)
        return arr if ncomp > 1 else arr[:, 0]
    return arr.reshape(count, ncomp) if ncomp > 1 else arr


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float32).reshape(4, 4).T
    T = np.eye(4, dtype=np.float32)
    if "translation" in node:
        T[:3, 3] = np.array(node["translation"], dtype=np.float32)
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )
        T[:3, :3] = T[:3, :3] @ R
    if "scale" in node:
        T[:3, :3] = T[:3, :3] @ np.diag(np.array(node["scale"], dtype=np.float32))
    return T


def _scene_nodes(gltf: dict) -> list[tuple[int, np.ndarray]]:
    nodes = gltf.get("nodes", [])
    if not nodes:
        return []
    scene_idx = gltf.get("scene", 0)
    root_ids = gltf.get("scenes", [{}])[scene_idx].get("nodes", list(range(len(nodes))))
    out: list[tuple[int, np.ndarray]] = []

    def visit(idx: int, parent: np.ndarray) -> None:
        node = nodes[idx]
        world = parent @ _node_matrix(node)
        out.append((idx, world))
        for child in node.get("children", []):
            visit(int(child), world)

    for root in root_ids:
        visit(int(root), np.eye(4, dtype=np.float32))
    return out


def _materials(gltf: dict) -> list[GltfMaterial]:
    out = []
    for mat in gltf.get("materials", []):
        pbr = mat.get("pbrMetallicRoughness", {})
        color = tuple(float(v) for v in pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]))
        tex = pbr.get("baseColorTexture", {}).get("index")
        out.append(
            GltfMaterial(
                name=mat.get("name"),
                base_color=color,  # type: ignore[arg-type]
                base_color_texture=int(tex) if tex is not None else None,
                metallic_factor=float(pbr.get("metallicFactor", 1.0)),
                roughness_factor=float(pbr.get("roughnessFactor", 1.0)),
            )
        )
    return out


def _image_bytes(gltf: dict, buffers: list[bytes], root_dir: str, image_idx: int) -> bytes:
    image = gltf["images"][image_idx]
    uri = image.get("uri")
    if uri is not None:
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        with open(os.path.join(root_dir, uri), "rb") as f:
            return f.read()
    view = gltf["bufferViews"][image["bufferView"]]
    raw = buffers[view["buffer"]]
    start = view.get("byteOffset", 0)
    end = start + view["byteLength"]
    return raw[start:end]


def _load_texture_image(gltf: dict, buffers: list[bytes], root_dir: str, texture_idx: int) -> mx.array:
    tex = gltf["textures"][texture_idx]
    img_idx = int(tex["source"])

    with Image.open(io.BytesIO(_image_bytes(gltf, buffers, root_dir, img_idx))) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return mx.array(arr)


def load_gltf(path: str) -> GltfData:
    """Load triangle mesh primitives from a ``.glb`` or ``.gltf`` file.

    The loader merges all triangle primitives referenced by the default scene
    into one indexed mesh and applies node transforms. Material indices are
    preserved per face when present.
    """
    if path.lower().endswith(".glb"):
        gltf, glb_bin = _read_glb(path)
    else:
        with open(path) as f:
            gltf = json.load(f)
        glb_bin = b""
    root_dir = os.path.dirname(os.path.abspath(path))
    buffers = _buffer_bytes(gltf, root_dir, glb_bin)
    materials = _materials(gltf)
    texture_ids = [m.base_color_texture for m in materials if m.base_color_texture is not None]
    texture_image = None
    if len(set(texture_ids)) == 1:
        texture_image = _load_texture_image(gltf, buffers, root_dir, texture_ids[0])

    verts_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray | None] = []
    uv_parts: list[np.ndarray | None] = []
    material_parts: list[np.ndarray] = []
    vert_offset = 0

    for node_idx, world in _scene_nodes(gltf):
        node = gltf["nodes"][node_idx]
        if "mesh" not in node:
            continue
        mesh = gltf["meshes"][node["mesh"]]
        normal_xform = np.linalg.inv(world[:3, :3]).T
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            attrs = prim["attributes"]
            verts = _read_accessor(gltf, buffers, attrs["POSITION"]).astype(np.float32)
            vh = np.concatenate([verts, np.ones((verts.shape[0], 1), dtype=np.float32)], axis=1)
            verts = (vh @ world.T)[:, :3].astype(np.float32)
            if "indices" in prim:
                faces = _read_accessor(gltf, buffers, prim["indices"]).astype(np.int32).reshape(-1, 3)
            else:
                faces = np.arange(verts.shape[0], dtype=np.int32).reshape(-1, 3)
            faces = faces + vert_offset

            normals = None
            if "NORMAL" in attrs:
                normals = _read_accessor(gltf, buffers, attrs["NORMAL"]).astype(np.float32)
                normals = (normals @ normal_xform.T).astype(np.float32)
                denom = np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
                normals = normals / denom
            uvs = (
                _read_accessor(gltf, buffers, attrs["TEXCOORD_0"]).astype(np.float32)
                if "TEXCOORD_0" in attrs
                else None
            )

            verts_parts.append(verts)
            faces_parts.append(faces)
            normal_parts.append(normals)
            uv_parts.append(uvs)
            material_parts.append(
                np.full((faces.shape[0],), int(prim.get("material", -1)), dtype=np.int32)
            )
            vert_offset += verts.shape[0]

    if not verts_parts:
        raise ValueError("No triangle mesh primitives found in glTF scene.")

    verts = np.concatenate(verts_parts, axis=0)
    faces = np.concatenate(faces_parts, axis=0)
    normals = (
        np.concatenate(
            [
                n if n is not None else np.zeros_like(v)
                for n, v in zip(normal_parts, verts_parts, strict=True)
            ],
            axis=0,
        )
        if any(n is not None for n in normal_parts)
        else None
    )
    uvs = (
        np.concatenate(
            [
                uv if uv is not None else np.zeros((v.shape[0], 2), dtype=np.float32)
                for uv, v in zip(uv_parts, verts_parts, strict=True)
            ],
            axis=0,
        )
        if any(uv is not None for uv in uv_parts)
        else None
    )
    material_ids = np.concatenate(material_parts, axis=0)
    return GltfData(
        verts=mx.array(verts),
        faces=mx.array(faces),
        normals=mx.array(normals) if normals is not None else None,
        uvs=mx.array(uvs) if uvs is not None else None,
        material_ids=mx.array(material_ids) if material_ids.size else None,
        materials=materials,
        texture_image=texture_image,
    )


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def _texture_png_bytes(texture_image: mx.array) -> bytes:
    tex = np.asarray(texture_image)
    if tex.ndim != 3 or tex.shape[-1] not in (3, 4):
        raise ValueError("texture_image must have shape (H, W, 3) or (H, W, 4).")
    if np.issubdtype(tex.dtype, np.floating):
        tex = np.clip(tex, 0.0, 1.0) * 255.0
    tex = np.asarray(tex, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(tex).save(buf, format="PNG")
    return buf.getvalue()


def save_gltf(
    path: str,
    verts: mx.array,
    faces: mx.array,
    normals: mx.array | None = None,
    uvs: mx.array | None = None,
    material_base_color: tuple[float, float, float, float] | None = None,
    texture_image: mx.array | None = None,
) -> None:
    """Save a triangle mesh as a self-contained binary ``.glb`` file."""
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
    n = np.asarray(normals, dtype=np.float32) if normals is not None else None
    uv = np.asarray(uvs, dtype=np.float32) if uvs is not None else None
    if texture_image is not None and uv is None:
        raise ValueError("uvs are required when saving texture_image.")

    blob = b""
    views, accessors, attributes = [], [], {}

    def _add(arr: np.ndarray, target: int, comp: int, typ: str, with_minmax: bool) -> int:
        nonlocal blob
        offset = len(blob)
        raw = arr.tobytes()
        blob += _pad4(raw)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        acc = {
            "bufferView": len(views) - 1,
            "componentType": comp,
            "count": int(arr.shape[0]),
            "type": typ,
        }
        if with_minmax:
            acc["min"] = arr.min(axis=0).tolist()
            acc["max"] = arr.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    attributes["POSITION"] = _add(v, 34962, 5126, "VEC3", with_minmax=True)
    if n is not None:
        attributes["NORMAL"] = _add(n, 34962, 5126, "VEC3", with_minmax=False)
    if uv is not None:
        attributes["TEXCOORD_0"] = _add(uv, 34962, 5126, "VEC2", with_minmax=False)
    idx_accessor = _add(f.reshape(-1), 34963, 5125, "SCALAR", with_minmax=False)
    texture_view = None
    if texture_image is not None:
        raw = _texture_png_bytes(texture_image)
        offset = len(blob)
        blob += _pad4(raw)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
        texture_view = len(views) - 1

    prim = {"attributes": attributes, "indices": idx_accessor, "mode": 4}
    materials = []
    if material_base_color is not None or texture_view is not None:
        pbr: dict[str, object] = {}
        if material_base_color is not None:
            pbr["baseColorFactor"] = [float(v) for v in material_base_color]
        if texture_view is not None:
            pbr["baseColorTexture"] = {"index": 0}
        materials.append({"pbrMetallicRoughness": pbr})
        prim["material"] = 0

    gltf = {
        "asset": {"version": "2.0", "generator": "mlx3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [prim]}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    if materials:
        gltf["materials"] = materials
    if texture_view is not None:
        gltf["images"] = [{"bufferView": texture_view, "mimeType": "image/png"}]
        gltf["textures"] = [{"source": 0}]

    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = _pad4(blob)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as out:
        out.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        out.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        out.write(json_chunk)
        out.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        out.write(bin_chunk)
