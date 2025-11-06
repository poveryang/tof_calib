"""Interactive tools for manually labeling chessboard corners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from .chessboard import ChessboardPattern, save_corners


@dataclass
class LabelingResult:
    image_path: Path
    points: np.ndarray


class InteractiveCornerLabeler:
    """Lightweight matplotlib-based corner labeler."""

    def __init__(self, pattern: ChessboardPattern, display_scale: float = 1.0):
        self.pattern = pattern
        self.display_scale = display_scale
        self._points: List[tuple[float, float]] = []
        self._figure = None
        self._axes = None
        self._image = None
        self._display_image = None
        self._done = False
        self._skip = False

    def _on_click(self, event) -> None:
        if event.inaxes != self._axes or event.button != 1:
            return
        if len(self._points) >= self.pattern.num_corners:
            return
        self._points.append((float(event.xdata), float(event.ydata)))
        self._refresh()

    def _on_key(self, event) -> None:
        if event.key == "escape" or event.key == "q":
            self._skip = True
            plt.close(self._figure)
        elif event.key in {"enter", " ", "tab"}:
            if len(self._points) == self.pattern.num_corners:
                self._done = True
                plt.close(self._figure)
        elif event.key in {"backspace", "delete"}:
            if self._points:
                self._points.pop()
                self._refresh()

    def _refresh(self) -> None:
        assert self._axes is not None
        self._axes.clear()
        self._axes.imshow(self._display_image, cmap="gray")
        self._axes.set_title(
            f"Corners: {len(self._points)}/{self.pattern.num_corners}"
            "  (Left click: add, Delete: undo, Enter: save, Q: skip)"
        )
        for idx, (x, y) in enumerate(self._points, start=1):
            self._axes.add_patch(Circle((x, y), 3, color="lime"))
            self._axes.text(x + 5, y - 5, str(idx), color="white", fontsize=8)
        self._axes.set_xlim(0, self._display_image.shape[1])
        self._axes.set_ylim(self._display_image.shape[0], 0)
        self._figure.canvas.draw_idle()

    def label(self, image_path: str | Path) -> LabelingResult | None:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        if abs(self.display_scale - 1.0) > 1e-3:
            h, w = gray.shape
            gray_display = cv2.resize(
                gray,
                (int(w * self.display_scale), int(h * self.display_scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            gray_display = gray

        self._points = []
        self._image = gray
        self._display_image = gray_display
        self._done = False
        self._skip = False

        plt.ioff()
        self._figure, self._axes = plt.subplots(figsize=(8, 8))
        self._figure.canvas.mpl_connect("button_press_event", self._on_click)
        self._figure.canvas.mpl_connect("key_press_event", self._on_key)
        self._refresh()
        plt.show(block=True)

        if self._skip or not self._points:
            return None

        points = np.array(self._points, dtype=np.float32)
        if abs(self.display_scale - 1.0) > 1e-3:
            points /= self.display_scale
        points = points[:, None, :]
        return LabelingResult(image_path=image_path, points=points)


def label_and_save(
    image_paths: List[str | Path],
    pattern: ChessboardPattern,
    output_dir: str | Path,
    display_scale: float = 1.0,
) -> list[LabelingResult]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labeler = InteractiveCornerLabeler(pattern, display_scale=display_scale)
    results: list[LabelingResult] = []

    for path in image_paths:
        result = labeler.label(path)
        if result is None:
            continue
        save_corners(output_dir / f"{Path(path).stem}_corners.npz", result.points)
        results.append(result)

    return results
