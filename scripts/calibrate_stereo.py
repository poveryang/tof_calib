#!/usr/bin/env python3
"""Stereo calibration between ToF and RGB cameras."""

from __future__ import annotations

import argparse
from pathlib import Path

from tof_calib import (
    ChessboardPattern,
    CornerDetectionConfig,
    calibrate_stereo,
    collect_corners,
    load_intrinsics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images_a", type=Path, help="Directory with camera A calibration images")
    parser.add_argument("images_b", type=Path, help="Directory with camera B calibration images")
    parser.add_argument("intrinsics_a", type=Path, help="Intrinsic calibration JSON for camera A")
    parser.add_argument("intrinsics_b", type=Path, help="Intrinsic calibration JSON for camera B")
    parser.add_argument("output", type=Path, help="Path to save the stereo calibration JSON")
    parser.add_argument("--columns", type=int, default=5, help="Number of inner corners along the chessboard width")
    parser.add_argument("--rows", type=int, default=4, help="Number of inner corners along the chessboard height")
    parser.add_argument("--square-size", type=float, default=30.0, help="Chessboard square size in millimetres")
    parser.add_argument("--corners-a", type=Path, default=None, help="Directory containing pre-computed corners for camera A")
    parser.add_argument("--corners-b", type=Path, default=None, help="Directory containing pre-computed corners for camera B")
    parser.add_argument("--save-corners-a", type=Path, default=None, help="Directory to save detected corners for camera A")
    parser.add_argument("--save-corners-b", type=Path, default=None, help="Directory to save detected corners for camera B")
    parser.add_argument("--vis-a", type=Path, default=None, help="Optional directory for visualising camera A detections")
    parser.add_argument("--vis-b", type=Path, default=None, help="Optional directory for visualising camera B detections")
    parser.add_argument("--no-equalize", dest="equalize", action="store_false", help="Disable CLAHE before detection")
    parser.add_argument("--blur", type=int, default=0, help="Gaussian blur kernel size (0 to disable)")
    parser.add_argument("--scale", type=float, default=1.0, help="Upscale factor before corner detection")
    parser.add_argument("--fast-check", action="store_true", help="Enable OpenCV fast-check optimisation")
    parser.add_argument("--no-subpix", dest="subpix", action="store_false", help="Disable sub-pixel refinement")
    return parser.parse_args()


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )


def main() -> None:
    args = parse_args()
    images_a = _list_images(args.images_a)
    images_b = _list_images(args.images_b)
    if len(images_a) != len(images_b):
        raise SystemExit("Camera A and B must provide the same number of images")
    if len(images_a) < 3:
        raise SystemExit("At least three image pairs are required")

    pattern = ChessboardPattern(args.columns, args.rows, args.square_size)
    config = CornerDetectionConfig(
        equalize_hist=args.equalize,
        blur_kernel=args.blur,
        scale=args.scale,
        use_fast_check=args.fast_check,
        refine_subpixel=args.subpix,
    )

    corners_a = collect_corners(
        images_a,
        pattern,
        config,
        existing_dir=args.corners_a,
        save_dir=args.save_corners_a,
        vis_dir=args.vis_a,
    )
    corners_b = collect_corners(
        images_b,
        pattern,
        config,
        existing_dir=args.corners_b,
        save_dir=args.save_corners_b,
        vis_dir=args.vis_b,
    )

    intrinsics_a = load_intrinsics(args.intrinsics_a)
    intrinsics_b = load_intrinsics(args.intrinsics_b)

    result = calibrate_stereo(pattern, corners_a, corners_b, intrinsics_a, intrinsics_b)
    result.save_json(args.output, intrinsics_a, intrinsics_b)
    print(f"Saved stereo calibration to {args.output}\nReprojection error: {result.reprojection_error:.4f} px")


if __name__ == "__main__":
    main()
