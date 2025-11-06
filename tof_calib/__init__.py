"""Core modules for the ToF/RGB calibration workflow."""

from .chessboard import (
    ChessboardPattern,
    CornerDetectionConfig,
    ChessboardCorners,
    detect_corners,
    draw_corners,
    load_corners,
    save_corners,
    collect_corners,
)
from .calibration import (
    IntrinsicCalibrationResult,
    StereoCalibrationResult,
    calibrate_intrinsics,
    calibrate_stereo,
    load_intrinsics,
)
from .projection import project_depth_to_rgb

__all__ = [
    "ChessboardPattern",
    "CornerDetectionConfig",
    "ChessboardCorners",
    "IntrinsicCalibrationResult",
    "StereoCalibrationResult",
    "detect_corners",
    "draw_corners",
    "load_corners",
    "save_corners",
    "collect_corners",
    "calibrate_intrinsics",
    "calibrate_stereo",
    "load_intrinsics",
    "project_depth_to_rgb",
]
