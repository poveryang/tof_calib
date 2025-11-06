#!/usr/bin/env python3
"""Monocular calibration using detected chessboard corners."""

from __future__ import annotations

import argparse
from pathlib import Path

from tof_calib import (
    ChessboardPattern,
    CornerDetectionConfig,
    calibrate_intrinsics,
    collect_corners,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Directory containing calibration images")
    parser.add_argument("output", type=Path, help="Path to save the calibration JSON")
    parser.add_argument("--columns", type=int, default=5, help="Number of inner corners along the chessboard width")
    parser.add_argument("--rows", type=int, default=4, help="Number of inner corners along the chessboard height")
    parser.add_argument("--square-size", type=float, default=30.0, help="Chessboard square size in millimetres")
    parser.add_argument("--corners", type=Path, default=None, help="Directory containing pre-computed corner files")
    parser.add_argument("--save-corners", type=Path, default=None, help="Directory to store detected corners")
    parser.add_argument("--vis", type=Path, default=None, help="Optional directory for visualising detections")
    parser.add_argument("--no-equalize", dest="equalize", action="store_false", help="Disable CLAHE before detection")
    parser.add_argument("--blur", type=int, default=0, help="Gaussian blur kernel size (0 to disable)")
    parser.add_argument("--scale", type=float, default=1.0, help="Upscale factor before corner detection")
    parser.add_argument("--fast-check", action="store_true", help="Enable OpenCV fast-check optimisation")
    parser.add_argument("--no-subpix", dest="subpix", action="store_false", help="Disable sub-pixel refinement")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if len(image_paths) < 3:
        raise SystemExit("At least three calibration images are required")

    pattern = ChessboardPattern(args.columns, args.rows, args.square_size)
    config = CornerDetectionConfig(
        equalize_hist=args.equalize,
        blur_kernel=args.blur,
        scale=args.scale,
        use_fast_check=args.fast_check,
        refine_subpixel=args.subpix,
    )

    corners = collect_corners(
        image_paths,
        pattern,
        config,
        existing_dir=args.corners,
        save_dir=args.save_corners,
        vis_dir=args.vis,
    )

    result = calibrate_intrinsics(corners, pattern)
    result.save_json(args.output)
    print(f"Saved calibration to {args.output}\nReprojection error: {result.reprojection_error:.4f} px")


if __name__ == "__main__":
    main()
