#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
交互式角点标注工具：
1. 逐张显示图像，用户手动点击标定角点（按顺序：行优先，左上到右下）
2. 保存每张图的角点文件
3. 可选：读取所有角点文件进行标定
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from calibrate_mono import build_object_points


class CornerLabeler:
    def __init__(
        self,
        image_path: str,
        inner_x: int,
        inner_y: int,
        scale: float = 1.0,
        window_name: str = "标注角点",
    ):
        self.image_path = image_path
        self.inner_x = inner_x
        self.inner_y = inner_y
        self.expected_count = inner_x * inner_y
        self.scale = scale
        self.window_name = window_name
        self.corners: List[Tuple[float, float]] = []
        self.img = None
        self.display_img = None
        self.fig = None
        self.ax = None
        self.done = False
        self.skip = False

    def on_click(self, event):
        """matplotlib 鼠标点击事件"""
        if event.inaxes != self.ax or event.button != 1:  # 左键
            return
        if len(self.corners) >= self.expected_count:
            return
        x, y = event.xdata, event.ydata
        self.corners.append((x, y))
        print(f"角点 {len(self.corners)}/{self.expected_count}: ({x:.1f}, {y:.1f})")
        self._update_display()

    def on_key(self, event):
        """键盘事件"""
        if event.key == "q":
            self.done = True
            plt.close(self.fig)
        elif event.key in ("enter", " "):
            # Enter 或空格保存当前已标注角点
            if len(self.corners) == self.expected_count:
                self.done = True
                plt.close(self.fig)
            else:
                print(f"角点数量不足（当前 {len(self.corners)}/{self.expected_count}）")
        elif event.key == "n":
            # 按 n：若已标满则自动保存并进入下一张；否则跳过
            if len(self.corners) == self.expected_count:
                self.done = True
            else:
                self.skip = True
            plt.close(self.fig)
        elif event.key == "backspace" or event.key == "delete":
            if self.corners:
                self.corners.pop()
                print(f"已撤销，剩余 {len(self.corners)} 个角点")
                self._update_display()

    def _update_display(self):
        """更新显示图像，绘制已标注的角点"""
        self.ax.clear()
        self.ax.imshow(self.display_img, cmap="gray" if self.display_img.ndim == 2 else None)
        self.ax.set_title(f"Corners: {len(self.corners)}/{self.expected_count} | Left-click: Add | Delete: Undo | Enter/Space: Save | n: Next | q: Quit")
        
        # 绘制已标注的角点
        for idx, (x, y) in enumerate(self.corners):
            color = "green" if idx == len(self.corners) - 1 else "red"
            circle = Circle((x, y), 3, color=color, fill=True)
            self.ax.add_patch(circle)
            self.ax.text(x + 5, y - 5, str(idx + 1), color="white", fontsize=8, weight="bold")

        # 连接相邻角点
        if len(self.corners) > 1:
            pts = np.array(self.corners)
            for i in range(1, len(pts)):
                self.ax.plot([pts[i-1][0], pts[i][0]], [pts[i-1][1], pts[i][1]], "g-", linewidth=1)

        self.ax.set_xlim(0, self.display_img.shape[1])
        self.ax.set_ylim(self.display_img.shape[0], 0)
        self.fig.canvas.draw()

    def label(self) -> Optional[np.ndarray]:
        """交互式标注角点，返回角点坐标（图像坐标系）"""
        self.img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if self.img is None:
            print(f"无法读取图像: {self.image_path}")
            return None

        # 转换为灰度显示
        if self.img.ndim == 3:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.img

        # 缩放显示
        if abs(self.scale - 1.0) > 1e-3:
            h, w = gray.shape[:2]
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            display_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            display_gray = gray

        self.display_img = display_gray
        self.corners = []
        self.done = False
        self.skip = False

        # 使用 matplotlib 交互式窗口
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        print(f"\n标注图像: {os.path.basename(self.image_path)}")
        print(f"预期角点数: {self.expected_count} (行优先，左上到右下)")
        print("操作: 左键点击添加角点 | Delete: 撤销 | 's': 保存 | 'n': 下一张 | 'q': 退出")

        self._update_display()
        plt.show()

        # 等待用户完成标注
        while not self.done and not self.skip:
            plt.pause(0.1)

        plt.close(self.fig)
        plt.ioff()

        if self.skip:
            print("跳过当前图像")
            return None

        if len(self.corners) != self.expected_count:
            print(f"角点数量不匹配（当前 {len(self.corners)}/{self.expected_count}）")
            return None

        # 将角点坐标还原到原图尺寸
        corners_scaled = []
        for x, y in self.corners:
            orig_x = x / self.scale if abs(self.scale - 1.0) > 1e-3 else x
            orig_y = y / self.scale if abs(self.scale - 1.0) > 1e-3 else y
            corners_scaled.append((orig_x, orig_y))

        # 转换为 (N, 1, 2) 格式
        corners_array = np.array(corners_scaled, dtype=np.float32).reshape(-1, 1, 2)
        return corners_array


def save_corners(corners: np.ndarray, output_path: str):
    """保存角点坐标到 .npz 文件"""
    np.savez_compressed(output_path, corners=corners)
    print(f"角点已保存: {output_path}")


def load_corners(corner_path: str) -> Optional[np.ndarray]:
    """加载角点文件"""
    if not os.path.exists(corner_path):
        return None
    data = np.load(corner_path)
    corners = data["corners"] if "corners" in data else data[data.files[0]]
    return corners.reshape(-1, 1, 2).astype(np.float32)


def label_images(
    images_dir: str,
    inner_x: int,
    inner_y: int,
    corners_dir: str,
    scale: float = 8.0,
    start_idx: int = 0,
    vis_dir: Optional[str] = None,
) -> List[str]:
    """标注一批图像的角点"""
    # 收集图像
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    image_paths: List[str] = []
    for ext in exts:
        image_paths.extend(Path(images_dir).glob(ext))
    image_paths = sorted([str(p) for p in image_paths])

    if not image_paths:
        print(f"目录中未找到图像: {images_dir}")
        return []

    os.makedirs(corners_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    labeled_files = []
    for idx, img_path in enumerate(image_paths[start_idx:], start=start_idx):
        labeler = CornerLabeler(img_path, inner_x, inner_y, scale=scale)
        corners = labeler.label()

        if corners is not None:
            base_name = Path(img_path).stem
            corner_path = os.path.join(corners_dir, f"{base_name}_corners.npz")
            save_corners(corners, corner_path)
            labeled_files.append(corner_path)
            
            # 保存角点可视化
            if vis_dir:
                from detect_chessboard_corners import draw_vis
                img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_raw is not None:
                    vis = draw_vis(img_raw, corners)
                    vis_path = os.path.join(vis_dir, f"{base_name}_vis.png")
                    cv2.imwrite(vis_path, vis)
                    print(f"可视化已保存: {vis_path}")
        else:
            print(f"跳过: {img_path}")

    return labeled_files


def calibrate_from_corner_files(
    images_dir: str,
    corners_dir: str,
    inner_x: int,
    inner_y: int,
    square_size_mm: float,
    output_json: str,
    visualize_dir: Optional[str] = None,
) -> None:
    """从角点文件标定相机内参"""
    # 收集图像和对应的角点文件
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    image_paths: List[str] = []
    for ext in exts:
        image_paths.extend(Path(images_dir).glob(ext))
    image_paths = sorted([str(p) for p in image_paths])

    corner_paths: List[str] = []
    valid_image_paths: List[str] = []

    for img_path in image_paths:
        base_name = Path(img_path).stem
        corner_path = os.path.join(corners_dir, f"{base_name}_corners.npz")
        if os.path.exists(corner_path):
            corner_paths.append(corner_path)
            valid_image_paths.append(img_path)

    if len(valid_image_paths) < 3:
        raise ValueError(f"有效图像数量不足（需要至少3张，当前 {len(valid_image_paths)} 张）")

    print(f"从 {len(valid_image_paths)} 张图像进行标定...")

    # 构建 object points 和 image points
    objp = build_object_points(inner_x, inner_y, square_size_mm)
    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None

    for img_path, corner_path in zip(valid_image_paths, corner_paths):
        corners = load_corners(corner_path)
        if corners is None or corners.shape[0] != inner_x * inner_y:
            print(f"跳过: {img_path} (角点数量不匹配)")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image_size is None and img is not None:
            h, w = img.shape[:2]
            image_size = (w, h)

        objpoints.append(objp)
        imgpoints.append(corners.astype(np.float32))

    if len(objpoints) < 3:
        raise ValueError(f"有效图像数量不足（需要至少3张，当前 {len(objpoints)} 张）")

    print(f"开始标定（有效图像 {len(objpoints)}）...")
    assert image_size is not None
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)

    result = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "inner_corners": [inner_x, inner_y],
        "square_size_mm": square_size_mm,
        "num_images": len(objpoints),
        "reprojection_error": float(mean_error),
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n标定完成！")
    print(f"平均重投影误差: {mean_error:.4f} 像素")
    print(f"相机内参矩阵 K:")
    print(mtx)
    print(f"畸变系数: {dist.flatten()}")
    print(f"结果已保存: {output_json}")

    # 可视化：生成去畸变前后的对比图
    if visualize_dir:
        os.makedirs(visualize_dir, exist_ok=True)
        print(f"\n生成可视化对比图...")
        # 对所有图像目录中的图像进行去畸变（不仅仅是标定用的）
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        all_images: List[str] = []
        for ext in exts:
            all_images.extend(Path(images_dir).glob(ext))
        all_images = sorted([str(p) for p in all_images])
        
        count = 0
        for img_path in all_images:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            h, w = img.shape[:2]
            newK, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
            undistorted = cv2.undistort(img, mtx, dist, None, newK)
            # 拼接对比（原图 | 去畸变）
            if img.ndim == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                und_bgr = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img
                und_bgr = undistorted
            comparison = np.hstack([img_bgr, und_bgr])
            out_path = os.path.join(visualize_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, comparison)
            count += 1
        print(f"可视化对比图已保存: {count} 张 → {visualize_dir}")


def main():
    parser = argparse.ArgumentParser(description="交互式角点标注工具")
    subparsers = parser.add_subparsers(dest="mode", help="模式")
    subparsers.required = True

    label_parser = subparsers.add_parser("label", help="标注模式：手动点击标注角点")
    label_parser.add_argument("--images-dir", required=True, type=str, help="棋盘图像目录")
    label_parser.add_argument("--corners-dir", required=True, type=str, help="角点文件保存目录")
    label_parser.add_argument("--inner-corners-x", type=int, default=5, help="水平内角点个数")
    label_parser.add_argument("--inner-corners-y", type=int, default=4, help="垂直内角点个数")
    label_parser.add_argument("--scale", type=float, default=8.0, help="显示缩放比例（默认8.0倍，便于精确点击）")
    label_parser.add_argument("--start-idx", type=int, default=0, help="从第几张开始标注（用于断点续标）")
    label_parser.add_argument("--save-vis", type=str, default=None, help="保存角点可视化的目录")

    calibrate_parser = subparsers.add_parser("calibrate", help="标定模式：从角点文件标定内参")
    calibrate_parser.add_argument("--images-dir", required=True, type=str, help="棋盘图像目录")
    calibrate_parser.add_argument("--corners-dir", required=True, type=str, help="角点文件保存目录")
    calibrate_parser.add_argument("--inner-corners-x", type=int, default=5, help="水平内角点个数")
    calibrate_parser.add_argument("--inner-corners-y", type=int, default=4, help="垂直内角点个数")
    calibrate_parser.add_argument("--square-size-mm", type=float, required=True, help="棋盘格单格边长（毫米）")
    calibrate_parser.add_argument("--output", type=str, default="manual_calibration.json", help="标定结果输出")
    calibrate_parser.add_argument("--visualize-dir", type=str, default=None, help="保存去畸变对比图的目录（可选）")

    args = parser.parse_args()

    if args.mode == "label":
        label_images(
            images_dir=args.images_dir,
            inner_x=args.inner_corners_x,
            inner_y=args.inner_corners_y,
            corners_dir=args.corners_dir,
            scale=args.scale,
            start_idx=args.start_idx,
            vis_dir=args.save_vis,
        )
    elif args.mode == "calibrate":
        calibrate_from_corner_files(
            images_dir=args.images_dir,
            corners_dir=args.corners_dir,
            inner_x=args.inner_corners_x,
            inner_y=args.inner_corners_y,
            square_size_mm=args.square_size_mm,
            output_json=args.output,
            visualize_dir=args.visualize_dir,
        )


if __name__ == "__main__":
    main()

