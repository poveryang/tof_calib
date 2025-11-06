#!/usr/bin/env python3
"""Batch chessboard corner detection for ToF/RGB calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from tof_calib import (
    ChessboardPattern,
    CornerDetectionConfig,
    collect_corners,
)


def _iter_images(directory: Path, extensions: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(sorted(directory.glob(f"*.{ext}")))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Directory containing calibration images")
    parser.add_argument("output", type=Path, help="Directory used to store detected corners")
    parser.add_argument("--columns", type=int, default=5, help="Number of inner corners along the chessboard width")
    parser.add_argument("--rows", type=int, default=4, help="Number of inner corners along the chessboard height")
    parser.add_argument("--square-size", type=float, default=30.0, help="Chessboard square size in millimetres")
    parser.add_argument("--no-equalize", dest="equalize", action="store_false", help="Disable CLAHE before detection")
    parser.add_argument("--blur", type=int, default=0, help="Gaussian blur kernel size (0 to disable)")
    parser.add_argument("--scale", type=float, default=1.0, help="Upscale factor before corner detection")
    parser.add_argument("--fast-check", action="store_true", help="Enable OpenCV fast-check optimisation")
    parser.add_argument("--no-subpix", dest="subpix", action="store_false", help="Disable sub-pixel refinement")
    parser.add_argument("--vis", type=Path, default=None, help="Optional directory for visualising detections")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = _iter_images(args.images, ["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    pattern = ChessboardPattern(args.columns, args.rows, args.square_size)
    config = CornerDetectionConfig(
        equalize_hist=args.equalize,
        blur_kernel=args.blur,
        scale=args.scale,
        use_fast_check=args.fast_check,
        refine_subpixel=args.subpix,
    )

    collect_corners(
        images,
        pattern,
        config,
        save_dir=args.output,
        vis_dir=args.vis,
    )
    print(f"Saved corners for {len(images)} images to {args.output}")


if __name__ == "__main__":
    main()
