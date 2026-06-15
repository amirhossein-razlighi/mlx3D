"""glTF 2.0 mesh IO (binary ``.glb`` and JSON ``.gltf``).

Loads the first triangle primitive of the first mesh (positions, indices, and
optional normals / UVs); saves a self-contained ``.glb``. glTF is the standard
interchange format for real-world assets and web viewers, so this unlocks
loading and exporting meshes that tools actually produce.

Coordinates are passed through unchanged (glTF is right-handed, +Y up).
"""

from __future__ import annotations

import base64
import json
import os
import struct
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

__all__ = ["GltfData", "load_gltf", "save_gltf"]

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
class GltfData:
    """Result of :func:`load_gltf`.

    Attributes:
        verts: ``(V, 3)`` positions.
        faces: ``(F, 3)`` triangle indices.
        normals: ``(V, 3)`` vertex normals, or ``None``.
        uvs: ``(V, 2)`` texture coordinates, or ``None``.
    """

    verts: mx.array
    faces: mx.array
    normals: mx.array | None = None
    uvs: mx.array | None = None


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
    arr = np.frombuffer(raw, dtype=dtype, count=count * ncomp, offset=start)
    return arr.reshape(count, ncomp) if ncomp > 1 else arr


def load_gltf(path: str) -> GltfData:
    """Load the first triangle mesh primitive from a ``.glb`` or ``.gltf`` file."""
    if path.lower().endswith(".glb"):
        gltf, glb_bin = _read_glb(path)
    else:
        with open(path) as f:
            gltf = json.load(f)
        glb_bin = b""
    buffers = _buffer_bytes(gltf, os.path.dirname(os.path.abspath(path)), glb_bin)

    prim = gltf["meshes"][0]["primitives"][0]
    if prim.get("mode", 4) != 4:
        raise ValueError("Only triangle primitives (mode 4) are supported.")
    attrs = prim["attributes"]
    verts = _read_accessor(gltf, buffers, attrs["POSITION"]).astype(np.float32)
    faces = _read_accessor(gltf, buffers, prim["indices"]).astype(np.int32).reshape(-1, 3)
    normals = (
        _read_accessor(gltf, buffers, attrs["NORMAL"]).astype(np.float32)
        if "NORMAL" in attrs
        else None
    )
    uvs = (
        _read_accessor(gltf, buffers, attrs["TEXCOORD_0"]).astype(np.float32)
        if "TEXCOORD_0" in attrs
        else None
    )
    return GltfData(
        verts=mx.array(verts),
        faces=mx.array(faces),
        normals=mx.array(normals) if normals is not None else None,
        uvs=mx.array(uvs) if uvs is not None else None,
    )


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def save_gltf(
    path: str,
    verts: mx.array,
    faces: mx.array,
    normals: mx.array | None = None,
) -> None:
    """Save a triangle mesh as a self-contained binary ``.glb`` file."""
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
    n = np.asarray(normals, dtype=np.float32) if normals is not None else None

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
    idx_accessor = _add(f.reshape(-1), 34963, 5125, "SCALAR", with_minmax=False)

    gltf = {
        "asset": {"version": "2.0", "generator": "mlx3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": attributes, "indices": idx_accessor, "mode": 4}]}
        ],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }

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
