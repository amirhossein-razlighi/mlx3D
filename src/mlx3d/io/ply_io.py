"""PLY loading and saving (ASCII and binary little-endian).

Supports the vertex properties used in practice — positions, normals,
colors, and arbitrary scalar properties (e.g. the ``f_dc_* / f_rest_* /
opacity / scale_* / rot_*`` properties of Gaussian Splatting checkpoints) —
plus triangular face lists.
"""

import os
from dataclasses import dataclass, field

import numpy as np

import mlx.core as mx

__all__ = ["load_ply", "save_ply", "PlyData"]

_PLY_TO_NP = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


@dataclass
class PlyData:
    """Result of :func:`load_ply`.

    Attributes:
        verts: (V, 3) positions.
        faces: (F, 3) triangle indices, or ``None`` for pure point clouds.
        normals: (V, 3) if ``nx, ny, nz`` present.
        colors: (V, 3) in [0, 1] if ``red, green, blue`` present.
        extra: dict of any remaining per-vertex scalar properties, each (V,).
    """

    verts: mx.array
    faces: mx.array | None = None
    normals: mx.array | None = None
    colors: mx.array | None = None
    extra: dict[str, mx.array] = field(default_factory=dict)


def load_ply(path: str) -> PlyData:
    """Load a PLY file (ascii or binary_little_endian)."""
    with open(path, "rb") as f:
        data = f.read()

    header_end = data.find(b"end_header\n")
    if header_end < 0:
        raise ValueError(f"{path!r} is not a valid PLY file (no end_header).")
    header = data[:header_end].decode("ascii", errors="replace").splitlines()
    body = data[header_end + len(b"end_header\n"):]

    if not header or header[0].strip() != "ply":
        raise ValueError(f"{path!r} is not a PLY file.")

    fmt = None
    elements: list[tuple[str, int, list]] = []  # (name, count, [(prop_name, dtype) or list-prop])
    for line in header[1:]:
        parts = line.strip().split()
        if not parts or parts[0] == "comment":
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if not elements:
                raise ValueError("property before element in PLY header.")
            if parts[1] == "list":
                elements[-1][2].append(("__list__", parts[4], _PLY_TO_NP[parts[2]], _PLY_TO_NP[parts[3]]))
            else:
                elements[-1][2].append((parts[2], _PLY_TO_NP[parts[1]]))

    if fmt not in ("ascii", "binary_little_endian"):
        raise ValueError(f"Unsupported PLY format {fmt!r}.")

    parsed: dict[str, dict[str, np.ndarray]] = {}
    offset = 0
    ascii_lines = body.decode("ascii", errors="replace").splitlines() if fmt == "ascii" else None
    ascii_pos = 0

    for name, count, props in elements:
        has_list = any(p[0] == "__list__" for p in props)
        if not has_list:
            dtype = np.dtype([(p[0], "<" + p[1]) for p in props])
            if fmt == "ascii":
                rows = []
                for _ in range(count):
                    rows.append(tuple(ascii_lines[ascii_pos].split()))
                    ascii_pos += 1
                arr = np.array(
                    [tuple(float(x) for x in r) for r in rows],
                    dtype=[(p[0], "f8") for p in props],
                ).astype(dtype)
            else:
                arr = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
                offset += dtype.itemsize * count
            parsed[name] = {p[0]: arr[p[0]] for p in props}
        else:
            # Element with a list property (faces). Assume one list property.
            lists = []
            if fmt == "ascii":
                for _ in range(count):
                    vals = ascii_lines[ascii_pos].split()
                    ascii_pos += 1
                    n = int(vals[0])
                    lists.append([int(x) for x in vals[1 : 1 + n]])
            else:
                lp = next(p for p in props if p[0] == "__list__")
                count_dt = np.dtype("<" + lp[2])
                index_dt = np.dtype("<" + lp[3])
                for _ in range(count):
                    n = int(np.frombuffer(body, dtype=count_dt, count=1, offset=offset)[0])
                    offset += count_dt.itemsize
                    idx = np.frombuffer(body, dtype=index_dt, count=n, offset=offset)
                    offset += index_dt.itemsize * n
                    lists.append(idx.tolist())
            # Fan-triangulate.
            tris = []
            for poly in lists:
                for i in range(1, len(poly) - 1):
                    tris.append([poly[0], poly[i], poly[i + 1]])
            parsed[name] = {"__faces__": np.asarray(tris, dtype=np.int32).reshape(-1, 3)}

    if "vertex" not in parsed:
        raise ValueError(f"No vertex element in {path!r}.")
    vert_props = parsed["vertex"]
    for axis in ("x", "y", "z"):
        if axis not in vert_props:
            raise ValueError(f"Vertex element missing {axis!r} property.")

    verts = np.stack(
        [vert_props["x"], vert_props["y"], vert_props["z"]], axis=-1
    ).astype(np.float32)
    consumed = {"x", "y", "z"}

    normals = None
    if all(k in vert_props for k in ("nx", "ny", "nz")):
        normals = np.stack(
            [vert_props["nx"], vert_props["ny"], vert_props["nz"]], axis=-1
        ).astype(np.float32)
        consumed |= {"nx", "ny", "nz"}

    colors = None
    if all(k in vert_props for k in ("red", "green", "blue")):
        rgb = np.stack(
            [vert_props["red"], vert_props["green"], vert_props["blue"]], axis=-1
        ).astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        colors = rgb
        consumed |= {"red", "green", "blue", "alpha"}

    extra = {
        k: mx.array(np.ascontiguousarray(v.astype(np.float32)))
        for k, v in vert_props.items()
        if k not in consumed
    }

    faces = None
    if "face" in parsed and parsed["face"]["__faces__"].size > 0:
        faces = mx.array(parsed["face"]["__faces__"])

    return PlyData(
        verts=mx.array(verts),
        faces=faces,
        normals=mx.array(normals) if normals is not None else None,
        colors=mx.array(colors) if colors is not None else None,
        extra=extra,
    )


def save_ply(
    path: str,
    verts: mx.array,
    faces: mx.array | None = None,
    normals: mx.array | None = None,
    colors: mx.array | None = None,
    extra: dict[str, mx.array] | None = None,
    binary: bool = True,
) -> None:
    """Save points/mesh to PLY.

    ``colors`` are expected in [0, 1] and stored as uchar. ``extra`` properties
    are stored as float32 in insertion order (useful for Gaussian Splatting
    checkpoints).
    """
    v = np.array(verts, dtype=np.float32)
    n = np.array(normals, dtype=np.float32) if normals is not None else None
    c = (
        np.clip(np.array(colors, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
        if colors is not None
        else None
    )
    extra_np = {k: np.array(a, dtype=np.float32).reshape(len(v)) for k, a in (extra or {}).items()}

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fmt = "binary_little_endian" if binary else "ascii"
    header = ["ply", f"format {fmt} 1.0", f"element vertex {len(v)}"]
    header += [f"property float {a}" for a in ("x", "y", "z")]
    if n is not None:
        header += [f"property float n{a}" for a in ("x", "y", "z")]
    if c is not None:
        header += [f"property uchar {a}" for a in ("red", "green", "blue")]
    header += [f"property float {k}" for k in extra_np]
    if faces is not None:
        header.append(f"element face {len(faces)}")
        header.append("property list uchar int vertex_indices")
    header.append("end_header")

    cols: list[np.ndarray] = [v]
    if n is not None:
        cols.append(n)
    if extra_np:
        pass  # appended after colors in struct order below

    with open(path, "wb") as out:
        out.write(("\n".join(header) + "\n").encode("ascii"))
        if binary:
            fields = [("xyz", "<f4", 3)]
            if n is not None:
                fields.append(("n", "<f4", 3))
            if c is not None:
                fields.append(("rgb", "u1", 3))
            for k in extra_np:
                fields.append((k, "<f4", 1))
            dtype = np.dtype([(name, dt, (cnt,)) for name, dt, cnt in fields])
            rec = np.empty(len(v), dtype=dtype)
            rec["xyz"] = v
            if n is not None:
                rec["n"] = n
            if c is not None:
                rec["rgb"] = c
            for k, a in extra_np.items():
                rec[k] = a[:, None]
            out.write(rec.tobytes())
            if faces is not None:
                f_arr = np.array(faces, dtype=np.int32)
                face_dtype = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])
                frec = np.empty(len(f_arr), dtype=face_dtype)
                frec["n"] = 3
                frec["idx"] = f_arr
                out.write(frec.tobytes())
        else:
            for i in range(len(v)):
                row = [f"{x:.8f}" for x in v[i]]
                if n is not None:
                    row += [f"{x:.8f}" for x in n[i]]
                if c is not None:
                    row += [str(int(x)) for x in c[i]]
                row += [f"{extra_np[k][i]:.8f}" for k in extra_np]
                out.write((" ".join(row) + "\n").encode("ascii"))
            if faces is not None:
                for tri in np.array(faces, dtype=np.int64):
                    out.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n".encode("ascii"))
