#!/usr/bin/env python3
"""Project a ToF depth CSV into the RGB image using calibrated parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tof_calib.projection import project_depth_to_rgb


def load_depth_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim != 2:
        raise ValueError("Depth CSV must describe a 2-D image")
    return data.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depth_csv", type=Path, help="Path to the depth CSV file")
    parser.add_argument("rgb_image", type=Path, help="Path to the RGB reference image")
    parser.add_argument("stereo_json", type=Path, help="Stereo calibration JSON output from calibrate_stereo.py")
    parser.add_argument("--depth-camera", choices=["a", "b"], default="a", help="Which camera in the stereo JSON corresponds to the depth sensor")
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Scale factor applied to the CSV values to convert to millimetres")
    parser.add_argument("--min-depth", type=float, default=None, help="Minimum depth value to keep after scaling")
    parser.add_argument("--max-depth", type=float, default=None, help="Maximum depth value to keep after scaling")
    parser.add_argument("--rotate-k", type=int, default=0, help="Rotate the depth image k times by 90 degrees")
    parser.add_argument("--pad-last-row", type=float, default=None, help="Append a row of this value to the bottom of the depth map")
    parser.add_argument("--output", type=Path, default=Path("projection.png"), help="Output path for the RGB overlay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depth = load_depth_csv(args.depth_csv)
    if args.pad_last_row is not None:
        depth = np.vstack([depth, np.full((1, depth.shape[1]), args.pad_last_row, dtype=np.float32)])
    if args.rotate_k:
        depth = np.rot90(depth, k=args.rotate_k)

    rgb = cv2.imread(str(args.rgb_image), cv2.IMREAD_COLOR)
    if rgb is None:
        raise SystemExit(f"Failed to read RGB image: {args.rgb_image}")

    with args.stereo_json.open("r", encoding="utf-8") as f:
        stereo = json.load(f)

    depth_key = args.depth_camera.lower()
    if depth_key == "a":
        K_depth = np.asarray(stereo["K_a"], dtype=np.float32)
        dist_depth = np.asarray(stereo["dist_a"], dtype=np.float32).reshape(-1, 1)
        K_rgb = np.asarray(stereo["K_b"], dtype=np.float32)
        dist_rgb = np.asarray(stereo["dist_b"], dtype=np.float32).reshape(-1, 1)
    else:
        K_depth = np.asarray(stereo["K_b"], dtype=np.float32)
        dist_depth = np.asarray(stereo["dist_b"], dtype=np.float32).reshape(-1, 1)
        K_rgb = np.asarray(stereo["K_a"], dtype=np.float32)
        dist_rgb = np.asarray(stereo["dist_a"], dtype=np.float32).reshape(-1, 1)

    R = np.asarray(stereo["R"], dtype=np.float32)
    t = np.asarray(stereo["t"], dtype=np.float32)

    result = project_depth_to_rgb(
        depth,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        K_depth=K_depth,
        dist_depth=dist_depth,
        K_rgb=K_rgb,
        dist_rgb=dist_rgb,
        rotation=R,
        translation=t,
        rgb_shape=rgb.shape[:2],
    )

    if result.points.size == 0:
        raise SystemExit("No valid points remained after projection")

    overlay = rgb.copy()
    depths = result.depths
    normalized = cv2.normalize(depths, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colors = cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

    for (x, y), color in zip(result.points.astype(int), colors):
        cv2.circle(overlay, (int(x), int(y)), 2, color.tolist(), -1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), overlay)
    print(
        f"Projected {len(result.points)} depth samples.\n"
        f"Output written to {args.output}\n"
        f"Depth range: {depths.min():.1f} - {depths.max():.1f} mm"
    )


if __name__ == "__main__":
    main()
