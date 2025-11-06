#!/usr/bin/env python3
"""Interactive manual labeling for ToF chessboard corners."""

from __future__ import annotations

import argparse
from pathlib import Path

from tof_calib import ChessboardPattern
from tof_calib.labeling import label_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Directory containing images to label")
    parser.add_argument("output", type=Path, help="Directory to store labelled corners")
    parser.add_argument("--columns", type=int, default=5, help="Number of inner corners along the chessboard width")
    parser.add_argument("--rows", type=int, default=4, help="Number of inner corners along the chessboard height")
    parser.add_argument("--square-size", type=float, default=30.0, help="Chessboard square size in millimetres")
    parser.add_argument("--scale", type=float, default=1.0, help="Display scale factor for the interactive window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    pattern = ChessboardPattern(args.columns, args.rows, args.square_size)
    label_and_save(image_paths, pattern, args.output, display_scale=args.scale)


if __name__ == "__main__":
    main()
