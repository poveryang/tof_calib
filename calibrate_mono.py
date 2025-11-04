#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
精简版单目内参标定：读取一批棋盘图 → 内置角点检测 → 标定内参。
"""

from __future__ import annotations

import argparse
import json
from typing import List, Tuple, Optional
import os
import glob

import cv2
import numpy as np

from detect_chessboard_corners import detect_corners, draw_vis


def build_object_points(inner_x: int, inner_y: int, square_size_mm: float) -> np.ndarray:
    """构建棋盘格的世界坐标（物理坐标）。
    
    Args:
        inner_x: 水平内角点数
        inner_y: 垂直内角点数
        square_size_mm: 单格边长（毫米）
    
    Returns:
        objectPoints: (N, 3) 数组，Z=0（棋盘在 XY 平面）
    """
    objp = np.zeros((inner_x * inner_y, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:inner_x, 0:inner_y].T.reshape(-1, 2)
    objp *= square_size_mm  # 转换为毫米
    return objp


def calibrate_mono(
    image_paths: List[str],
    inner_x: int,
    inner_y: int,
    square_size_mm: float,
    eq_hist: bool = True,
    blur_ksize: int = 0,
    prefer_sb: bool = True,
    scale: float = 1.0,
    do_subpix: bool = True,
    subpix_win: int = 3,
    save_vis_dir: Optional[str] = None,
    vis_scale: float = 4.0,
    corners_dir: Optional[str] = None,
) -> Tuple[dict, float]:
    """单目相机标定。
    
    Args:
        image_paths: 图像文件路径列表
        inner_x: 水平内角点数
        inner_y: 垂直内角点数
        square_size_mm: 棋盘格单格边长（毫米）
        corners_dir: 角点文件目录（如果提供，优先使用已有的角点文件）
        其他参数: 角点检测参数（仅在未找到角点文件时使用）
    
    Returns:
        (标定结果字典, 平均重投影误差)
    """
    objp = build_object_points(inner_x, inner_y, square_size_mm)
    
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    print(f"共 {len(image_paths)} 张图，开始加载/检测角点...")
    if corners_dir:
        print(f"  优先从目录加载角点: {corners_dir}")
    image_size: Optional[Tuple[int, int]] = None
    loaded_count = 0
    detected_count = 0
    
    for img_path in image_paths:
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        corners = None
        raw = None
        
        # 优先尝试从文件加载角点
        if corners_dir:
            corner_path = os.path.join(corners_dir, f"{img_basename}_corners.npz")
            if os.path.exists(corner_path):
                try:
                    data = np.load(corner_path)
                    # 兼容不同的键名
                    if "corners" in data:
                        corners = data["corners"]
                    else:
                        # 如果没有 "corners" 键，使用第一个数组
                        corners = data[data.files[0]]
                    # 确保形状为 (N, 1, 2)
                    if corners.ndim == 2:
                        corners = corners[:, None, :]
                    corners = corners.astype(np.float32)
                    if corners.shape[0] == inner_x * inner_y:
                        # 加载原始图像以获取尺寸
                        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if raw is None:
                            raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
                            if raw is not None:
                                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                        if raw is None:
                            print(f"  ⚠ 跳过（无法读取图像）: {img_path}")
                            continue
                        loaded_count += 1
                        print(f"  ✓ [加载] {img_basename}")
                    else:
                        corners = None  # 角点数量不匹配，重新检测
                except Exception as e:
                    print(f"  ⚠ 加载角点失败 {corner_path}: {e}，将重新检测")
                    corners = None
        
        # 如果没有加载到角点，进行自动检测
        if corners is None:
            found, corners, raw = detect_corners(
                image_path=img_path,
                inner_x=inner_x,
                inner_y=inner_y,
                eq_hist=eq_hist,
                blur_ksize=blur_ksize,
                invert=False,
                scale=scale,
                roi=None,
                use_fast_check=False,
                prefer_sb=prefer_sb,
                debug=False,
                do_subpix=do_subpix,
                subpix_win=subpix_win,
            )
            if not found or corners is None or corners.shape[0] != inner_x * inner_y:
                print(f"  ✗ 跳过（检测失败）: {img_path}")
                continue
            detected_count += 1
            print(f"  ✓ [检测] {img_basename}")
        
        objpoints.append(objp)
        imgpoints.append(corners.astype(np.float32))
        if image_size is None:
            h, w = raw.shape[:2]
            image_size = (w, h)
        
        # 可选：保存每张图的角点可视化
        if save_vis_dir:
            os.makedirs(save_vis_dir, exist_ok=True)
            vis = draw_vis(raw, corners)
            if vis_scale and abs(vis_scale - 1.0) > 1e-3:
                nh, nw = int(round(vis.shape[0] * vis_scale)), int(round(vis.shape[1] * vis_scale))
                vis = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_NEAREST)
            out_path = os.path.join(save_vis_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, vis)
    
    if loaded_count > 0 or detected_count > 0:
        print(f"\n  角点统计: 加载 {loaded_count} 个，检测 {detected_count} 个")
    
    if len(objpoints) < 3:
        raise ValueError(f"有效图像数量不足（需要至少3张，当前 {len(objpoints)} 张）")
    
    print(f"开始标定（有效图像 {len(objpoints)}）...")
    
    # 执行标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    
    # 计算重投影误差
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
    
    print(f"完成。重投影误差: {mean_error:.4f} 像素")
    print(f"相机内参矩阵 K:")
    print(mtx)
    print(f"畸变系数: {dist.flatten()}")
    
    return result, mean_error


def main():
    parser = argparse.ArgumentParser(description="单目内参标定")
    parser.add_argument("--images-dir", required=True, type=str, help="棋盘图像目录（自动匹配常见后缀）")
    parser.add_argument("--corners-dir", type=str, default=None, help="角点文件目录（如果提供，优先使用已有的角点文件）")
    parser.add_argument("--inner-corners-x", type=int, default=5, help="水平内角点个数（默认 5）")
    parser.add_argument("--inner-corners-y", type=int, default=4, help="垂直内角点个数（默认 4）")
    parser.add_argument("--square-size-mm", type=float, default=40.0, help="棋盘格单格边长（毫米，默认 40.0）")
    parser.add_argument("--eq-hist", action="store_true")
    parser.add_argument("--blur", type=int, default=0)
    parser.add_argument("--sb", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--no-subpix", action="store_true")
    parser.add_argument("--subpix-win", type=int, default=3)
    parser.add_argument("--output", type=str, default="calibration_result.json")
    parser.add_argument("--save-vis-dir", type=str, default=None, help="保存每张图角点可视化的目录")
    parser.add_argument("--vis-scale", type=float, default=4.0, help="可视化输出放大倍数（默认4.0）")

    args = parser.parse_args()

    # 收集目录下的图像
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    image_paths: List[str] = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(args.images_dir, ext)))
    image_paths = sorted(image_paths)
    if not image_paths:
        raise ValueError(f"目录中未找到图像: {args.images_dir}")

    result, reproj_error = calibrate_mono(
        image_paths=image_paths,
        inner_x=args.inner_corners_x,
        inner_y=args.inner_corners_y,
        square_size_mm=args.square_size_mm,
        eq_hist=args.eq_hist,
        blur_ksize=args.blur,
        scale=args.scale,
        prefer_sb=args.sb,
        do_subpix=(not args.no_subpix),
        subpix_win=args.subpix_win,
        save_vis_dir=args.save_vis_dir,
        vis_scale=args.vis_scale,
        corners_dir=args.corners_dir,
    )

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n标定结果已保存: {args.output}")


if __name__ == "__main__":
    main()

