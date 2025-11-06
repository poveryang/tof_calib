"""Utilities for projecting ToF depth maps into the RGB camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class ProjectionResult:
    points: np.ndarray
    depths: np.ndarray
    mask: np.ndarray


def project_depth_to_rgb(
    depth: np.ndarray,
    *,
    depth_scale: float,
    min_depth: Optional[float],
    max_depth: Optional[float],
    K_depth: np.ndarray,
    dist_depth: np.ndarray,
    K_rgb: np.ndarray,
    dist_rgb: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    rgb_shape: tuple[int, int],
    min_positive_z: float = 1.0,
) -> ProjectionResult:
    """Project a depth image into the RGB camera frame."""

    if depth.ndim != 2:
        raise ValueError("Depth input must be a 2-D array")

    scaled = depth.astype(np.float32) * float(depth_scale)
    mask = np.isfinite(scaled)
    if min_depth is not None:
        mask &= scaled >= float(min_depth)
    if max_depth is not None:
        mask &= scaled <= float(max_depth)

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return ProjectionResult(points=np.empty((0, 2), dtype=np.float32), depths=np.empty((0,), dtype=np.float32), mask=np.zeros_like(depth, dtype=bool))

    depth_valid = scaled[ys, xs]
    pixels = np.column_stack((xs.astype(np.float32), ys.astype(np.float32))).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pixels, K_depth, dist_depth).reshape(-1, 2)

    points_depth = np.zeros((len(depth_valid), 3), dtype=np.float32)
    points_depth[:, 0] = normalized[:, 0] * depth_valid
    points_depth[:, 1] = normalized[:, 1] * depth_valid
    points_depth[:, 2] = depth_valid

    R = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float32).reshape(3, 1)

    points_rgb = (R @ points_depth.T + t).T
    positive = points_rgb[:, 2] > float(min_positive_z)
    points_rgb = points_rgb[positive]
    depth_valid = depth_valid[positive]
    mask_projected = np.zeros_like(depth, dtype=bool)
    mask_projected[ys[positive], xs[positive]] = True

    if len(points_rgb) == 0:
        return ProjectionResult(points=np.empty((0, 2), dtype=np.float32), depths=np.empty((0,), dtype=np.float32), mask=mask_projected)

    proj, _ = cv2.projectPoints(
        points_rgb.reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        K_rgb,
        dist_rgb,
    )
    proj = proj.reshape(-1, 2)

    h_rgb, w_rgb = rgb_shape
    inside = (
        (proj[:, 0] >= 0)
        & (proj[:, 0] < w_rgb)
        & (proj[:, 1] >= 0)
        & (proj[:, 1] < h_rgb)
    )

    return ProjectionResult(
        points=proj[inside].astype(np.float32),
        depths=depth_valid[inside].astype(np.float32),
        mask=mask_projected,
    )
