"""Built-in incremental Structure-from-Motion (no COLMAP required).

A compact, classical SfM pipeline in the spirit of COLMAP, sized for casual
captures (tens to a few hundred images of one scene):

1. SIFT features + ratio-test matching (exhaustive for small sets, sequential
   window + log-spaced loop-closure pairs for ordered/video captures).
2. Two-view initialization from the verified pair with the most inliers among
   those with a healthy triangulation angle.
3. Incremental registration with PnP-RANSAC, triangulation of new tracks, and
   periodic global bundle adjustment (sparse Levenberg-Marquardt via SciPy,
   robust soft-L1 loss, shared-focal refinement).

Poses/intrinsics use the OpenCV/COLMAP convention throughout and results are
written as a standard COLMAP binary sparse model, so downstream code cannot
tell it apart from a real COLMAP reconstruction. Requires the optional
``[capture]`` extra (OpenCV + SciPy).

For hard captures (wide baselines, low texture, loops) install COLMAP -- the
pipeline prefers it automatically when available. Remaining pose error is
absorbed by joint pose refinement during 3DGS training (``--refine-poses``).
"""

import math
import os
from dataclasses import dataclass

import numpy as np

from ..cameras import Camera
from .frames import estimate_focal_px

__all__ = ["SfmConfig", "SfmResult", "run_sfm"]


def _import_cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The built-in SfM needs OpenCV and SciPy. Install the capture extra:\n"
            '    pip install "mlx3d[capture]"\n'
            "or install COLMAP (e.g. `brew install colmap`) to use it instead."
        ) from e
    return cv2


@dataclass
class SfmConfig:
    max_dim: int = 1024
    """Images are downscaled to this size for feature extraction/matching."""
    n_features: int = 4096
    match_ratio: float = 0.75
    min_pair_inliers: int = 30
    sequential_window: int = 8
    exhaustive_threshold: int = 60
    """Match all pairs when there are at most this many images."""
    ransac_px: float = 2.0
    pnp_px: float = 6.0
    min_pnp_inliers: int = 12
    min_tri_angle_deg: float = 1.0
    init_min_tri_angle_deg: float = 3.0
    max_reproj_px: float = 4.0
    ba_every: int = 8
    """Run global bundle adjustment after this many new registrations."""
    refine_focal: bool = True
    seed: int = 0


@dataclass
class SfmResult:
    sparse_dir: str
    registered: list[str]
    num_points: int
    mean_reproj_px: float


# ------------------------------------------------------------------ geometry
def _rotvecs_to_matrices(rvecs: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(rvecs.reshape(-1, 3)).as_matrix()


def _project(xyz_cam: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    z = np.maximum(xyz_cam[:, 2], 1e-8)
    return np.stack([f * xyz_cam[:, 0] / z + cx, f * xyz_cam[:, 1] / z + cy], axis=1)


def _triangulation_angles(p3d: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    d1 = p3d - c1
    d2 = p3d - c2
    d1 /= np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-12)
    d2 /= np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-12)
    cos = np.clip((d1 * d2).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


class _Reconstruction:
    """Incremental SfM state: keypoints, tracks, poses, and 3D points."""

    def __init__(self, keypoints, colors, K, cfg: SfmConfig):
        self.kp = keypoints  # per image: (Ni, 2) float64 pixel coords
        self.kp_colors = colors  # per image: (Ni, 3) float in [0, 1]
        self.K = K
        self.cfg = cfg
        n = len(keypoints)
        self.poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # img -> (R, t)
        self.feat_to_point: list[dict[int, int]] = [dict() for _ in range(n)]
        self.points: list[np.ndarray] = []
        self.point_obs: list[dict[int, int]] = []  # pid -> {img: feat}

    # -- basic accessors
    def center(self, i: int) -> np.ndarray:
        R, t = self.poses[i]
        return -R.T @ t

    def add_point(self, xyz: np.ndarray, obs: dict[int, int]) -> int:
        pid = len(self.points)
        self.points.append(xyz)
        self.point_obs.append(dict(obs))
        for img, feat in obs.items():
            self.feat_to_point[img][feat] = pid
        return pid

    def add_observation(self, pid: int, img: int, feat: int) -> None:
        if feat in self.feat_to_point[img]:
            return
        self.point_obs[pid][img] = feat
        self.feat_to_point[img][feat] = pid

    def reproj_errors(self, pid: int) -> float:
        xyz = self.points[pid]
        errs = []
        for img, feat in self.point_obs[pid].items():
            R, t = self.poses[img]
            uv = _project((R @ xyz + t)[None], self.K[0, 0], self.K[0, 2], self.K[1, 2])[0]
            errs.append(float(np.linalg.norm(uv - self.kp[img][feat])))
        return max(errs) if errs else 0.0

    # -- bundle adjustment
    def bundle_adjust(self, max_nfev: int = 25, log=print) -> None:
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix
        from scipy.spatial.transform import Rotation

        imgs = sorted(self.poses)
        if len(imgs) < 2 or not self.points:
            return
        img_index = {img: k for k, img in enumerate(imgs)}
        obs = []  # (cam_idx, pid, u, v)
        pid_used = sorted(pid for pid in range(len(self.points)) if len(self.point_obs[pid]) >= 2)
        pid_index = {pid: k for k, pid in enumerate(pid_used)}
        for pid in pid_used:
            for img, feat in self.point_obs[pid].items():
                obs.append((img_index[img], pid_index[pid], *self.kp[img][feat]))
        if not obs:
            return
        cam_idx = np.array([o[0] for o in obs], dtype=np.int64)
        pt_idx = np.array([o[1] for o in obs], dtype=np.int64)
        uv = np.array([(o[2], o[3]) for o in obs])

        n_cams, n_pts = len(imgs), len(pid_used)
        refine_f = self.cfg.refine_focal
        # Layout: [f?] + 6 dof per camera except camera 0 (gauge fix) + 3 per point.
        n_f = 1 if refine_f else 0
        pose0 = []
        for img in imgs:
            R, t = self.poses[img]
            pose0.append(np.concatenate([Rotation.from_matrix(R).as_rotvec(), t]))
        pose0 = np.array(pose0)
        x0 = np.concatenate(
            [
                np.array([self.K[0, 0]]) if refine_f else np.empty(0),
                pose0[1:].ravel(),
                np.array([self.points[p] for p in pid_used]).ravel(),
            ]
        )
        fixed_pose = pose0[0]
        cx, cy = self.K[0, 2], self.K[1, 2]

        def unpack(x):
            f = x[0] if refine_f else self.K[0, 0]
            poses = np.concatenate(
                [fixed_pose[None], x[n_f : n_f + 6 * (n_cams - 1)].reshape(-1, 6)]
            )
            pts = x[n_f + 6 * (n_cams - 1) :].reshape(-1, 3)
            return f, poses, pts

        def residuals(x):
            f, poses, pts = unpack(x)
            Rs = _rotvecs_to_matrices(poses[:, :3])
            p = pts[pt_idx]
            xyz_cam = np.einsum("nij,nj->ni", Rs[cam_idx], p) + poses[cam_idx, 3:]
            return (_project(xyz_cam, f, cx, cy) - uv).ravel()

        m = 2 * len(obs)
        A = lil_matrix((m, x0.size), dtype=np.int8)
        rows = np.arange(len(obs))
        if refine_f:
            A[:, 0] = 1
        for k in range(6):
            cols = n_f + 6 * (cam_idx - 1) + k
            valid = cam_idx > 0
            A[2 * rows[valid], cols[valid]] = 1
            A[2 * rows[valid] + 1, cols[valid]] = 1
        for k in range(3):
            cols = n_f + 6 * (n_cams - 1) + 3 * pt_idx + k
            A[2 * rows, cols] = 1
            A[2 * rows + 1, cols] = 1

        res = least_squares(
            residuals,
            x0,
            jac_sparsity=A,
            x_scale="jac",
            ftol=1e-4,
            method="trf",
            loss="soft_l1",
            f_scale=2.0,
            max_nfev=max_nfev,
        )
        f, poses, pts = unpack(res.x)
        if refine_f:
            self.K[0, 0] = self.K[1, 1] = float(f)
        Rs = _rotvecs_to_matrices(poses[:, :3])
        for k, img in enumerate(imgs):
            self.poses[img] = (Rs[k], poses[k, 3:])
        for k, pid in enumerate(pid_used):
            self.points[pid] = pts[k]
        rms = math.sqrt(float(np.mean(res.fun**2))) * math.sqrt(2.0)
        log(f"    BA: {n_cams} cams, {n_pts} pts, {len(obs)} obs, ~{rms:.2f} px")

    def filter_points(self) -> int:
        """Drop points with large reprojection error or too few observations."""
        removed = 0
        for pid in range(len(self.points)):
            obs = self.point_obs[pid]
            if not obs:
                continue
            if len(obs) < 2 or self.reproj_errors(pid) > self.cfg.max_reproj_px:
                for img, feat in obs.items():
                    self.feat_to_point[img].pop(feat, None)
                self.point_obs[pid] = {}
                removed += 1
        return removed


def _match_pair(matcher, desc1, desc2, ratio: float):
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    return np.array([(m.queryIdx, m.trainIdx) for m in good], dtype=np.int64).reshape(-1, 2)


def _candidate_pairs(n: int, cfg: SfmConfig) -> list[tuple[int, int]]:
    if n <= cfg.exhaustive_threshold:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs = set()
    for i in range(n):
        for d in range(1, cfg.sequential_window + 1):
            if i + d < n:
                pairs.add((i, i + d))
        d = cfg.sequential_window * 2
        while i + d < n:  # log-spaced long-range pairs for loop closure
            pairs.add((i, i + d))
            d *= 2
    return sorted(pairs)


def run_sfm(
    image_paths: list[str],
    out_root: str,
    config: SfmConfig | None = None,
    log=print,
) -> SfmResult:
    """Estimate camera poses + sparse points for ``image_paths``.

    Writes a COLMAP binary sparse model under ``out_root/sparse/0`` (only
    registered images are included) and returns an :class:`SfmResult`.
    """
    cv2 = _import_cv2()
    from ..datasets import save_colmap

    cfg = config or SfmConfig()
    if len(image_paths) < 3:
        raise ValueError("SfM needs at least 3 images.")
    cv2.setRNGSeed(cfg.seed)

    # ---- load (downscaled) images, shared intrinsics prior
    from PIL import Image

    with Image.open(image_paths[0]) as im0:
        orig_w, orig_h = im0.size
    scale = min(1.0, cfg.max_dim / max(orig_w, orig_h))
    w, h = round(orig_w * scale), round(orig_h * scale)
    focal_prior, focal_source = estimate_focal_px(image_paths[0])
    f = focal_prior * scale
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    log(
        f"  {len(image_paths)} images at {w}x{h} (SfM scale {scale:.3f}), "
        f"focal prior {focal_prior:.0f} px ({focal_source})"
    )

    sift = cv2.SIFT_create(nfeatures=cfg.n_features)
    keypoints, descriptors, colors = [], [], []
    for p in image_paths:
        with Image.open(p) as img:
            if img.size != (orig_w, orig_h):
                raise ValueError(
                    "The built-in SfM assumes one camera: all images must share "
                    f"one size, but {os.path.basename(p)} is {img.size} vs "
                    f"{(orig_w, orig_h)}. Use COLMAP for mixed captures."
                )
            rgb = np.asarray(img.convert("RGB").resize((w, h), Image.BILINEAR))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(kps) < 10:
            kps, desc = [], np.zeros((0, 128), dtype=np.float32)
        pts = np.array([kp.pt for kp in kps], dtype=np.float64).reshape(-1, 2)
        px = np.clip(pts.round().astype(int), 0, [w - 1, h - 1])
        keypoints.append(pts)
        descriptors.append(desc)
        colors.append(rgb[px[:, 1], px[:, 0]].astype(np.float64) / 255.0)
    log(f"  SIFT: median {int(np.median([len(k) for k in keypoints]))} features/image")

    # ---- pairwise matching + geometric verification
    pairs = _candidate_pairs(len(image_paths), cfg)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches: dict[tuple[int, int], np.ndarray] = {}
    for i, j in pairs:
        if min(len(keypoints[i]), len(keypoints[j])) < cfg.min_pair_inliers:
            continue
        m = _match_pair(matcher, descriptors[i], descriptors[j], cfg.match_ratio)
        if len(m) < cfg.min_pair_inliers:
            continue
        p1, p2 = keypoints[i][m[:, 0]], keypoints[j][m[:, 1]]
        E, inl = cv2.findEssentialMat(
            p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=cfg.ransac_px
        )
        if E is None or inl is None:
            continue
        m = m[inl.ravel().astype(bool)]
        if len(m) >= cfg.min_pair_inliers:
            matches[(i, j)] = m
    log(f"  Verified {len(matches)} image pairs (of {len(pairs)} candidates)")
    if not matches:
        raise RuntimeError(
            "No image pairs with enough matches. The images likely do not overlap or lack texture."
        )

    rec = _Reconstruction(keypoints, colors, K, cfg)

    # ---- initialization: best verified pair with a healthy triangulation angle
    def _init_pair_stats(i, j, m):
        p1, p2 = keypoints[i][m[:, 0]], keypoints[j][m[:, 1]]
        E, inl = cv2.findEssentialMat(
            p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=cfg.ransac_px
        )
        if E is None:
            return None
        ok = inl.ravel().astype(bool)
        n_pose, R, t, mask = cv2.recoverPose(E, p1[ok], p2[ok], K)
        if n_pose < cfg.min_pair_inliers:
            return None
        good = mask.ravel().astype(bool)
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        X = cv2.triangulatePoints(P1, P2, p1[ok][good].T, p2[ok][good].T)
        X = (X[:3] / np.maximum(np.abs(X[3]), 1e-12) * np.sign(X[3])).T
        angles = _triangulation_angles(X, np.zeros(3), (-R.T @ t).ravel())
        return R, t.ravel(), m[ok][good], X, float(np.median(angles)), int(n_pose)

    best = None
    for (i, j), m in sorted(matches.items(), key=lambda kv: -len(kv[1]))[:40]:
        stats = _init_pair_stats(i, j, m)
        if stats is None:
            continue
        R, t, m_good, X, med_angle, n_pose = stats
        score = n_pose * min(1.0, med_angle / cfg.init_min_tri_angle_deg) ** 2
        if best is None or score > best[0]:
            best = (score, i, j, R, t, m_good, X, med_angle)
    if best is None:
        raise RuntimeError("Could not find a valid initial image pair.")
    _, i0, j0, R1, t1, m_good, X, med_angle = best
    log(
        f"  Init pair ({os.path.basename(image_paths[i0])}, "
        f"{os.path.basename(image_paths[j0])}): {len(m_good)} points, "
        f"median angle {med_angle:.1f} deg"
    )
    rec.poses[i0] = (np.eye(3), np.zeros(3))
    rec.poses[j0] = (R1, t1)
    keep = X[:, 2] > 0
    for xyz, (f1, f2) in zip(X[keep], m_good[keep]):
        rec.add_point(xyz, {i0: int(f1), j0: int(f2)})

    # ---- incremental registration
    def correspondences_2d3d(img: int):
        obj, pix, feats = [], [], []
        for (a, b), m in matches.items():
            if a == img and b in rec.poses:
                new_f, reg_i, reg_f = m[:, 0], b, m[:, 1]
            elif b == img and a in rec.poses:
                new_f, reg_i, reg_f = m[:, 1], a, m[:, 0]
            else:
                continue
            for nf, rf in zip(new_f, reg_f):
                pid = rec.feat_to_point[reg_i].get(int(rf))
                if pid is not None and rec.point_obs[pid]:
                    obj.append(rec.points[pid])
                    pix.append(keypoints[img][nf])
                    feats.append((int(nf), pid))
        return np.array(obj), np.array(pix), feats

    def triangulate_new(img: int) -> int:
        added = 0
        R_new, t_new = rec.poses[img]
        P_new = K @ np.hstack([R_new, t_new[:, None]])
        c_new = rec.center(img)
        for (a, b), m in matches.items():
            if a == img and b in rec.poses:
                other, f_img, f_oth = b, m[:, 0], m[:, 1]
            elif b == img and a in rec.poses:
                other, f_img, f_oth = a, m[:, 1], m[:, 0]
            else:
                continue
            R_o, t_o = rec.poses[other]
            P_o = K @ np.hstack([R_o, t_o[:, None]])
            c_o = rec.center(other)
            for fi, fo in zip(f_img, f_oth):
                fi, fo = int(fi), int(fo)
                pid_i = rec.feat_to_point[img].get(fi)
                pid_o = rec.feat_to_point[other].get(fo)
                if pid_i is not None and pid_o is None:
                    rec.add_observation(pid_i, other, fo)
                    continue
                if pid_o is not None and pid_i is None:
                    rec.add_observation(pid_o, img, fi)
                    continue
                if pid_i is not None or pid_o is not None:
                    continue
                X4 = cv2.triangulatePoints(
                    P_new, P_o, keypoints[img][fi][:, None], keypoints[other][fo][:, None]
                )
                if abs(X4[3, 0]) < 1e-12:
                    continue
                xyz = (X4[:3, 0] / X4[3, 0]).ravel()
                z1 = (R_new @ xyz + t_new)[2]
                z2 = (R_o @ xyz + t_o)[2]
                if z1 <= 0 or z2 <= 0:
                    continue
                uv1 = _project((R_new @ xyz + t_new)[None], K[0, 0], K[0, 2], K[1, 2])[0]
                uv2 = _project((R_o @ xyz + t_o)[None], K[0, 0], K[0, 2], K[1, 2])[0]
                if (
                    np.linalg.norm(uv1 - keypoints[img][fi]) > cfg.max_reproj_px
                    or np.linalg.norm(uv2 - keypoints[other][fo]) > cfg.max_reproj_px
                ):
                    continue
                angle = _triangulation_angles(xyz[None], c_new, c_o)[0]
                if angle < cfg.min_tri_angle_deg:
                    continue
                rec.add_point(xyz, {img: fi, other: fo})
                added += 1
        return added

    triangulate_new(i0)  # extend the initial pair's tracks to shared neighbors
    failed: set[int] = set()
    since_ba = 0
    while True:
        remaining = [i for i in range(len(image_paths)) if i not in rec.poses and i not in failed]
        if not remaining:
            break
        scored = []
        for img in remaining:
            obj, pix, feats = correspondences_2d3d(img)
            if len(obj) >= cfg.min_pnp_inliers:
                scored.append((len(obj), img, obj, pix, feats))
        if not scored:
            break
        _, img, obj, pix, feats = max(scored, key=lambda s: s[0])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj.reshape(-1, 1, 3),
            pix.reshape(-1, 1, 2),
            K,
            None,
            reprojectionError=cfg.pnp_px,
            confidence=0.999,
            iterationsCount=500,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inl is None or len(inl) < cfg.min_pnp_inliers:
            failed.add(img)
            continue
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(
            obj[inl].reshape(-1, 1, 3), pix[inl].reshape(-1, 1, 2), K, None, rvec, tvec
        )
        R = cv2.Rodrigues(rvec)[0]
        rec.poses[img] = (R, tvec.ravel())
        for k in inl:
            feat, pid = feats[k]
            rec.add_observation(pid, img, feat)
        n_new = triangulate_new(img)
        log(
            f"  Registered {os.path.basename(image_paths[img])}: "
            f"{len(inl)} PnP inliers, +{n_new} points "
            f"({len(rec.poses)}/{len(image_paths)} images)"
        )
        since_ba += 1
        if since_ba >= cfg.ba_every:
            rec.bundle_adjust(max_nfev=20, log=log)
            rec.filter_points()
            since_ba = 0

    log("  Final bundle adjustment...")
    rec.bundle_adjust(max_nfev=75, log=log)
    removed = rec.filter_points()
    if removed:
        log(f"  Filtered {removed} unstable points")

    registered = sorted(rec.poses)
    if len(registered) < 3:
        raise RuntimeError(
            f"Only {len(registered)} of {len(image_paths)} images registered; "
            "not enough for training. Try COLMAP, more overlap, or sharper images."
        )
    skipped = len(image_paths) - len(registered)
    if skipped:
        log(f"  Warning: {skipped} images could not be registered and were skipped.")

    # ---- export at the original resolution
    import mlx.core as mx

    inv_scale = 1.0 / scale
    f_full = rec.K[0, 0] * inv_scale
    cams, names = [], []
    for img in registered:
        R, t = rec.poses[img]
        cams.append(
            Camera(
                R=mx.array(R.astype(np.float32)),
                t=mx.array(t.astype(np.float32)),
                fx=f_full,
                fy=f_full,
                cx=rec.K[0, 2] * inv_scale,
                cy=rec.K[1, 2] * inv_scale,
                width=orig_w,
                height=orig_h,
            )
        )
        names.append(os.path.basename(image_paths[img]))

    pids = [pid for pid in range(len(rec.points)) if len(rec.point_obs[pid]) >= 2]
    xyz = np.array([rec.points[p] for p in pids]).reshape(-1, 3)
    col = np.empty((len(pids), 3))
    for k, p in enumerate(pids):
        img = min(rec.point_obs[p])  # color from the first observing image
        col[k] = rec.kp_colors[img][rec.point_obs[p][img]]
    errors = np.array([rec.reproj_errors(p) for p in pids])
    sparse_dir = save_colmap(out_root, cams, names, xyz, col, point_errors=errors)
    mean_err = float(errors.mean()) if len(errors) else 0.0
    log(
        f"  SfM done: {len(registered)} cameras, {len(pids)} points, "
        f"mean reprojection {mean_err:.2f} px, focal {f_full:.0f} px"
    )
    return SfmResult(
        sparse_dir=sparse_dir,
        registered=names,
        num_points=len(pids),
        mean_reproj_px=mean_err,
    )
