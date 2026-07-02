"""Frame ingestion for the capture pipeline: photos directory or video file.

Videos are sampled with ffmpeg, then filtered by sharpness so motion-blurred
frames (the main quality killer for hand-held captures) never reach pose
estimation: the video is oversampled ~2.5x above the requested frame count and
the sharpest frame of each time bucket is kept (variance-of-Laplacian score, a
standard blur measure).
"""

import math
import os
import shutil
import subprocess

import numpy as np

__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "list_images",
    "is_video",
    "sharpness_score",
    "select_sharpest",
    "extract_video_frames",
    "estimate_focal_px",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def is_video(path: str) -> bool:
    """Whether ``path`` looks like a video file the pipeline can ingest."""
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def list_images(directory: str) -> list[str]:
    """Sorted image files (by name) directly inside ``directory``."""
    names = [
        n
        for n in sorted(os.listdir(directory))
        if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS
    ]
    return [os.path.join(directory, n) for n in names]


def _load_gray(path: str, max_dim: int = 640) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("L")
        scale = max(img.width, img.height) / max_dim
        if scale > 1:
            img = img.resize((round(img.width / scale), round(img.height / scale)))
        return np.asarray(img, dtype=np.float32)


def sharpness_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian; higher means sharper. ``gray`` is (H, W) float."""
    lap = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]
    )
    return float(lap.var())


def select_sharpest(paths: list[str], target: int) -> list[str]:
    """Pick ``target`` frames from a temporally ordered list of frame paths.

    The sequence is split into ``target`` contiguous buckets and the sharpest
    frame of each bucket is kept, preserving even temporal coverage while
    dropping blurred frames.
    """
    if target <= 0:
        raise ValueError("target must be positive.")
    if len(paths) <= target:
        return list(paths)
    scores = [sharpness_score(_load_gray(p)) for p in paths]
    keep: list[str] = []
    for b in range(target):
        lo = round(b * len(paths) / target)
        hi = round((b + 1) * len(paths) / target)
        best = max(range(lo, hi), key=lambda i: scores[i])
        keep.append(paths[best])
    return keep


def _ffmpeg_or_raise() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "ffmpeg is required to extract frames from a video but was not found "
            "on PATH. Install it (e.g. `brew install ffmpeg`) or pass a directory "
            "of photos instead."
        )
    return exe


def _video_duration_seconds(video: str) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError):
        return None


def extract_video_frames(
    video: str,
    out_dir: str,
    num_frames: int = 150,
    overscan: float = 2.5,
    jpeg_quality: int = 2,
    log=print,
) -> list[str]:
    """Extract ``num_frames`` sharp frames from ``video`` into ``out_dir``.

    ffmpeg samples the video evenly at ``overscan * num_frames`` frames, then
    :func:`select_sharpest` keeps the sharpest frame per time bucket and the
    rejects are deleted. Returns the kept frame paths (sorted, temporal order).
    """
    ffmpeg = _ffmpeg_or_raise()
    os.makedirs(out_dir, exist_ok=True)
    raw = int(math.ceil(num_frames * max(overscan, 1.0)))
    duration = _video_duration_seconds(video)
    if duration and duration > 0:
        fps = raw / duration
        vf = f"fps={fps:.6f}"
        log(f"Sampling {video} at {fps:.2f} fps (~{raw} frames from {duration:.1f}s)...")
    else:
        # Unknown duration (no ffprobe): sample at a fixed rate.
        vf = "fps=4"
        log(f"Sampling {video} at 4 fps (unknown duration)...")
    pattern = os.path.join(out_dir, "frame_%05d.jpg")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            video,
            "-vf",
            vf,
            "-q:v",
            str(jpeg_quality),
            pattern,
        ],
        check=True,
    )
    extracted = list_images(out_dir)
    if not extracted:
        raise RuntimeError(f"ffmpeg extracted no frames from {video!r}.")
    keep = set(select_sharpest(extracted, num_frames))
    for p in extracted:
        if p not in keep:
            os.remove(p)
    kept = list_images(out_dir)
    log(f"Kept {len(kept)} sharp frames of {len(extracted)} sampled.")
    return kept


def estimate_focal_px(image_path: str, prior: float = 1.2) -> tuple[float, str]:
    """Estimate the focal length in pixels for an image.

    Uses the EXIF 35mm-equivalent focal length when present (mapping the 36mm
    film width onto the image's long side), otherwise falls back to
    ``prior * max(width, height)`` — the same default prior COLMAP uses.

    Returns ``(focal_px, source)`` with ``source`` in ``{"exif", "prior"}``.
    """
    from PIL import ExifTags, Image

    with Image.open(image_path) as img:
        w, h = img.size
        try:
            exif = img.getexif()
            f35 = exif.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.FocalLengthIn35mmFilm)
        except Exception:
            f35 = None
    if f35:
        return float(f35) / 36.0 * max(w, h), "exif"
    return prior * max(w, h), "prior"
