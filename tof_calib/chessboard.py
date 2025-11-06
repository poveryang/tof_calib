"""Chessboard corner utilities used across the calibration workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ChessboardPattern:
    """Definition of a planar chessboard calibration target."""

    columns: int
    rows: int
    square_size_mm: float

    @property
    def size(self) -> tuple[int, int]:
        return self.columns, self.rows

    @property
    def num_corners(self) -> int:
        return self.columns * self.rows

    def object_points(self) -> np.ndarray:
        grid = np.mgrid[0 : self.columns, 0 : self.rows].T.reshape(-1, 2)
        obj = np.zeros((self.num_corners, 3), dtype=np.float32)
        obj[:, :2] = grid * float(self.square_size_mm)
        return obj


@dataclass(frozen=True)
class CornerDetectionConfig:
    """Parameters controlling chessboard corner detection."""

    equalize_hist: bool = True
    blur_kernel: int = 0
    invert: bool = False
    scale: float = 1.0
    roi: tuple[int, int, int, int] | None = None
    use_fast_check: bool = False
    prefer_sb: bool = True
    refine_subpixel: bool = True
    subpixel_window: int = 5


@dataclass
class ChessboardCorners:
    """Detected chessboard corners in image coordinates."""

    image_path: Path
    points: np.ndarray
    image_size: tuple[int, int]

    def as_float32(self) -> np.ndarray:
        pts = np.asarray(self.points, dtype=np.float32)
        if pts.ndim == 2:
            pts = pts[:, None, :]
        return pts


# ---------------------------------------------------------------------------
# Corner detection helpers
# ---------------------------------------------------------------------------

def _to_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    lo, hi = np.percentile(gray.astype(np.float32), (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(np.min(gray)), float(np.max(gray))
        if hi <= lo:
            return np.zeros_like(gray, dtype=np.uint8)
    scaled = (gray.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _preprocess(gray: np.ndarray, config: CornerDetectionConfig) -> np.ndarray:
    image = _to_uint8(gray)
    if config.invert:
        image = cv2.bitwise_not(image)
    if config.equalize_hist:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    if config.blur_kernel > 0:
        k = max(1, int(config.blur_kernel) | 1)
        image = cv2.GaussianBlur(image, (k, k), 0)
    return image


def _apply_roi(gray: np.ndarray, roi: tuple[int, int, int, int] | None) -> tuple[np.ndarray, tuple[int, int]]:
    if roi is None:
        return gray, (0, 0)
    x, y, w, h = roi
    H, W = gray.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x1 + max(0, w)), min(H, y1 + max(0, h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI is empty or out of bounds")
    return gray[y1:y2, x1:x2], (x1, y1)


def _resize_if_needed(image: np.ndarray, scale: float) -> tuple[np.ndarray, float]:
    if abs(scale - 1.0) < 1e-3:
        return image, 1.0
    scale = float(scale)
    new_w = max(1, int(round(image.shape[1] * scale)))
    new_h = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized, scale


def detect_corners(
    image_path: str | Path,
    pattern: ChessboardPattern,
    config: CornerDetectionConfig,
) -> ChessboardCorners:
    """Detect chessboard corners from a single image."""

    image_path = Path(image_path)
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
    roi_view, roi_offset = _apply_roi(gray, config.roi)
    preprocessed = _preprocess(roi_view, config)
    resized, scale_used = _resize_if_needed(preprocessed, config.scale)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if config.use_fast_check:
        flags |= cv2.CALIB_CB_FAST_CHECK

    pattern_size = pattern.size
    found = False
    corners = None

    if config.prefer_sb and hasattr(cv2, "findChessboardCornersSB"):
        try:
            sb_result = cv2.findChessboardCornersSB(resized, pattern_size)
            if isinstance(sb_result, tuple):
                found, corners = bool(sb_result[0]), sb_result[1]
            else:
                corners = sb_result
                found = corners is not None
        except cv2.error:
            found = False
            corners = None

    if not found:
        found, corners = cv2.findChessboardCorners(resized, pattern_size, flags)

    if not found or corners is None:
        raise RuntimeError(f"Failed to detect {pattern.num_corners} corners in {image_path}")

    corners = np.asarray(corners, dtype=np.float32)
    if corners.ndim == 2:
        corners = corners[:, None, :]

    if config.refine_subpixel:
        win = max(1, int(config.subpixel_window))
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.01,
        )
        cv2.cornerSubPix(resized, corners, (win, win), (-1, -1), criteria)

    if scale_used != 1.0:
        corners /= scale_used
    corners += np.array(roi_offset, dtype=np.float32)

    return ChessboardCorners(
        image_path=image_path,
        points=corners.astype(np.float32),
        image_size=(gray.shape[1], gray.shape[0]),
    )


def draw_corners(image: np.ndarray, corners: np.ndarray, pattern: ChessboardPattern) -> np.ndarray:
    vis = image.copy()
    if image.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, pattern.size, corners, True)
    return vis


# ---------------------------------------------------------------------------
# Corner persistence
# ---------------------------------------------------------------------------

def save_corners(path: str | Path, corners: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, corners=np.asarray(corners, dtype=np.float32))


def load_corners(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path)
    key = "corners" if "corners" in data else data.files[0]
    arr = np.asarray(data[key], dtype=np.float32)
    return arr[:, None, :] if arr.ndim == 2 else arr


def collect_corners(
    image_paths: Sequence[str | Path],
    pattern: ChessboardPattern,
    config: CornerDetectionConfig,
    *,
    existing_dir: str | Path | None = None,
    save_dir: str | Path | None = None,
    vis_dir: str | Path | None = None,
) -> list[ChessboardCorners]:
    results: list[ChessboardCorners] = []
    existing_dir = Path(existing_dir) if existing_dir else None
    save_dir = Path(save_dir) if save_dir else None
    vis_dir = Path(vis_dir) if vis_dir else None

    for path in image_paths:
        path = Path(path)
        base = path.stem
        loaded: Optional[np.ndarray] = None
        if existing_dir:
            loaded = load_corners(existing_dir / f"{base}_corners.npz")
        if loaded is not None and loaded.shape[0] == pattern.num_corners:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(path)
            h, w = image.shape[:2]
            record = ChessboardCorners(path, loaded.astype(np.float32), (w, h))
            results.append(record)
            continue

        record = detect_corners(path, pattern, config)
        results.append(record)

        if save_dir:
            save_corners(save_dir / f"{base}_corners.npz", record.as_float32())

        if vis_dir:
            vis_dir.mkdir(parents=True, exist_ok=True)
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(path)
            vis = draw_corners(raw, record.as_float32(), pattern)
            cv2.imwrite(str(vis_dir / f"{base}.png"), vis)

    return results
