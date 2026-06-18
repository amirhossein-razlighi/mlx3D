"""Compact Gaussian Splatting checkpoints."""

from __future__ import annotations

import argparse
import json

from ..splatting import GaussianModel


def compact_checkpoint(
    input_path: str,
    output_path: str,
    min_opacity: float = 0.0,
    max_gaussians: int | None = None,
    sh_degree: int | None = None,
) -> dict[str, object]:
    """Compact a 3DGS PLY checkpoint and return summary stats."""
    model = GaussianModel.load_ply(input_path)
    before = model.num_gaussians
    compact = model.compact(
        min_opacity=min_opacity,
        max_gaussians=max_gaussians,
        target_sh_degree=sh_degree,
    )
    compact.save_ply(output_path)
    return {
        "input": input_path,
        "output": output_path,
        "gaussians_before": before,
        "gaussians_after": compact.num_gaussians,
        "sh_degree_before": model.sh_degree,
        "sh_degree_after": compact.sh_degree,
        "min_opacity": float(min_opacity),
        "max_gaussians": max_gaussians,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="input 3DGS PLY checkpoint")
    parser.add_argument("--out", required=True, help="output compacted PLY checkpoint")
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--sh-degree", type=int, default=None, help="truncate to this SH degree")
    parser.add_argument("--json-out", help="optional JSON summary path")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_opacity < 0.0 or args.min_opacity > 1.0:
        parser.error("--min-opacity must be in [0, 1].")
    if args.max_gaussians is not None and args.max_gaussians <= 0:
        parser.error("--max-gaussians must be positive.")
    if args.sh_degree is not None and args.sh_degree < 0:
        parser.error("--sh-degree must be non-negative.")

    summary = compact_checkpoint(
        args.input,
        args.out,
        min_opacity=args.min_opacity,
        max_gaussians=args.max_gaussians,
        sh_degree=args.sh_degree,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
