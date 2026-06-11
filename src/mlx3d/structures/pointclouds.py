"""A batched point-cloud container with list / packed / padded views."""

import mlx.core as mx

__all__ = ["Pointclouds"]


def _as_float_array(a) -> mx.array:
    a = mx.array(a) if not isinstance(a, mx.array) else a
    return a.astype(mx.float32) if mx.issubdtype(a.dtype, mx.integer) else a


class Pointclouds:
    """Batch of point clouds with optional per-point normals and features.

    Args:
        points: list of ``(P_i, 3)`` arrays or a padded ``(N, P, 3)`` array.
        normals: optional, same layout as ``points``.
        features: optional, list of ``(P_i, C)`` arrays or padded ``(N, P, C)``.
    """

    def __init__(self, points, normals=None, features=None) -> None:
        if isinstance(points, mx.array):
            if points.ndim != 3 or points.shape[-1] != 3:
                raise ValueError("Padded points must have shape (N, P, 3).")
            points = [points[i] for i in range(points.shape[0])]
            if isinstance(normals, mx.array):
                normals = [normals[i] for i in range(normals.shape[0])]
            if isinstance(features, mx.array):
                features = [features[i] for i in range(features.shape[0])]
        if not isinstance(points, (list, tuple)):
            raise ValueError("points must be a list of arrays or a padded mx.array.")

        self._points_list = [_as_float_array(p) for p in points]
        for p in self._points_list:
            if p.ndim != 2 or p.shape[-1] != 3:
                raise ValueError("Each points entry must have shape (P, 3).")

        def _check_aux(aux, name):
            if aux is None:
                return None
            aux = [_as_float_array(a) for a in aux]
            if len(aux) != len(self._points_list):
                raise ValueError(f"{name} must have one entry per cloud.")
            for a, p in zip(aux, self._points_list):
                if a.shape[0] != p.shape[0]:
                    raise ValueError(f"{name} entries must match points per cloud.")
            return aux

        self._normals_list = _check_aux(normals, "normals")
        self._features_list = _check_aux(features, "features")
        self._N = len(self._points_list)

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, index) -> "Pointclouds":
        if isinstance(index, int):
            index = [index]
        if isinstance(index, slice):
            index = list(range(self._N))[index]
        return Pointclouds(
            [self._points_list[i] for i in index],
            normals=[self._normals_list[i] for i in index] if self._normals_list else None,
            features=[self._features_list[i] for i in index] if self._features_list else None,
        )

    @property
    def num_points_per_cloud(self) -> mx.array:
        return mx.array([p.shape[0] for p in self._points_list], dtype=mx.int32)

    # -------------------------------------------------------------------- views
    def points_list(self) -> list[mx.array]:
        return self._points_list

    def normals_list(self) -> list[mx.array] | None:
        return self._normals_list

    def features_list(self) -> list[mx.array] | None:
        return self._features_list

    def points_packed(self) -> mx.array:
        return (
            mx.concatenate(self._points_list, axis=0)
            if self._N
            else mx.zeros((0, 3))
        )

    def normals_packed(self) -> mx.array | None:
        if self._normals_list is None:
            return None
        return mx.concatenate(self._normals_list, axis=0)

    def features_packed(self) -> mx.array | None:
        if self._features_list is None:
            return None
        return mx.concatenate(self._features_list, axis=0)

    def points_packed_to_cloud_idx(self) -> mx.array:
        return mx.concatenate(
            [
                mx.full((p.shape[0],), i, dtype=mx.int32)
                for i, p in enumerate(self._points_list)
            ]
            or [mx.zeros((0,), dtype=mx.int32)]
        )

    def points_padded(self) -> mx.array:
        P = max((p.shape[0] for p in self._points_list), default=0)
        rows = []
        for p in self._points_list:
            pad = P - p.shape[0]
            rows.append(mx.pad(p, ((0, pad), (0, 0))) if pad > 0 else p)
        return mx.stack(rows, axis=0) if rows else mx.zeros((0, 0, 3))

    def padded_mask(self) -> mx.array:
        """(N, max(P_i)) boolean mask of valid points in ``points_padded``."""
        P = max((p.shape[0] for p in self._points_list), default=0)
        idx = mx.arange(P)[None, :]
        return idx < self.num_points_per_cloud[:, None]

    # ------------------------------------------------------------- modification
    def offset_points(self, offsets: mx.array) -> "Pointclouds":
        """Add ``offsets`` (packed ``(sum(P_i), 3)`` or broadcastable) to all points."""
        new_points = self.points_packed() + offsets
        out = []
        offset = 0
        for p in self._points_list:
            out.append(new_points[offset : offset + p.shape[0]])
            offset += p.shape[0]
        return Pointclouds(out, normals=self._normals_list, features=self._features_list)

    def scale_points(self, scale) -> "Pointclouds":
        return Pointclouds(
            [p * scale for p in self._points_list],
            normals=self._normals_list,
            features=self._features_list,
        )
