"""Calibration helpers for the ToF/RGB toolchain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import json

import cv2
import numpy as np

from .chessboard import ChessboardCorners, ChessboardPattern


@dataclass
class IntrinsicCalibrationResult:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    reprojection_error: float

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion.reshape(-1, 1).tolist(),
            "image_width": int(self.image_size[0]),
            "image_height": int(self.image_size[1]),
            "reprojection_error": float(self.reprojection_error),
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class StereoCalibrationResult:
    rotation: np.ndarray
    translation: np.ndarray
    essential: np.ndarray
    fundamental: np.ndarray
    reprojection_error: float

    def to_dict(self, intrinsics_a: IntrinsicCalibrationResult, intrinsics_b: IntrinsicCalibrationResult) -> dict:
        return {
            "K_a": intrinsics_a.camera_matrix.tolist(),
            "dist_a": intrinsics_a.distortion.reshape(-1, 1).tolist(),
            "K_b": intrinsics_b.camera_matrix.tolist(),
            "dist_b": intrinsics_b.distortion.reshape(-1, 1).tolist(),
            "R": self.rotation.tolist(),
            "t": self.translation.reshape(-1, 1).tolist(),
            "E": self.essential.tolist(),
            "F": self.fundamental.tolist(),
            "reprojection_error": float(self.reprojection_error),
        }

    def save_json(
        self,
        path: str | Path,
        intrinsics_a: IntrinsicCalibrationResult,
        intrinsics_b: IntrinsicCalibrationResult,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(intrinsics_a, intrinsics_b), f, indent=2)


def _validate_sizes(corners: Sequence[ChessboardCorners]) -> tuple[int, int]:
    image_sizes = {c.image_size for c in corners}
    if len(image_sizes) != 1:
        raise ValueError("All images must share the same resolution for calibration")
    return next(iter(image_sizes))


def calibrate_intrinsics(
    corners: Sequence[ChessboardCorners],
    pattern: ChessboardPattern,
) -> IntrinsicCalibrationResult:
    if len(corners) < 3:
        raise ValueError("At least three images are required for intrinsic calibration")

    image_size = _validate_sizes(corners)
    obj_points = [pattern.object_points() for _ in corners]
    img_points = [c.as_float32() for c in corners]

    error, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
    )

    total_err = 0.0
    total_points = 0
    for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
        err = cv2.norm(img, proj, cv2.NORM_L2)
        total_err += err
        total_points += len(obj)

    mean_err = total_err / max(total_points, 1)

    return IntrinsicCalibrationResult(
        camera_matrix=camera_matrix,
        distortion=distortion,
        image_size=image_size,
        reprojection_error=float(mean_err),
    )


def calibrate_stereo(
    pattern: ChessboardPattern,
    corners_a: Sequence[ChessboardCorners],
    corners_b: Sequence[ChessboardCorners],
    intrinsics_a: IntrinsicCalibrationResult,
    intrinsics_b: IntrinsicCalibrationResult,
    flags: int = cv2.CALIB_FIX_INTRINSIC,
) -> StereoCalibrationResult:
    if len(corners_a) != len(corners_b):
        raise ValueError("Corner lists must have the same length")
    if len(corners_a) < 3:
        raise ValueError("At least three image pairs are required for stereo calibration")

    size_a = _validate_sizes(corners_a)
    size_b = _validate_sizes(corners_b)
    if size_a != size_b:
        raise ValueError("Image sizes for the two cameras must match")

    obj_points = [pattern.object_points() for _ in corners_a]
    img_points_a = [c.as_float32() for c in corners_a]
    img_points_b = [c.as_float32() for c in corners_b]

    error, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_a,
        img_points_b,
        intrinsics_a.camera_matrix,
        intrinsics_a.distortion,
        intrinsics_b.camera_matrix,
        intrinsics_b.distortion,
        size_a,
        flags=flags,
    )

    return StereoCalibrationResult(
        rotation=R,
        translation=T.reshape(3, 1),
        essential=E,
        fundamental=F,
        reprojection_error=float(error),
    )


def load_intrinsics(path: str | Path) -> IntrinsicCalibrationResult:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data.get("distortion_coefficients") or data.get("dist_coeffs"), dtype=np.float64)
    width = int(data.get("image_width"))
    height = int(data.get("image_height"))
    reprojection_error = float(data.get("reprojection_error", 0.0))

    return IntrinsicCalibrationResult(
        camera_matrix=camera_matrix,
        distortion=dist.reshape(-1, 1),
        image_size=(width, height),
        reprojection_error=reprojection_error,
    )
