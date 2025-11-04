#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动检测 RGB 图像的棋盘角点并保存为 npz 格式（与 manual_corner_labeling.py 输出一致）。

使用方法：
  python3 auto_detect_rgb_corners.py \
    --images-dir /path/to/rgb_images \
    --corners-dir /path/to/rgb_corners \
    --inner-corners-x 6 --inner-corners-y 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detect_chessboard_corners import detect_corners


def save_corners(corners: np.ndarray, output_path: Path):
    """保存角点坐标到 .npz 文件（与 manual_corner_labeling.py 格式一致）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), corners=corners)
    print(f"✓ 角点已保存: {output_path}")


def process_images(
    images_dir: Path,
    corners_dir: Path,
    inner_x: int,
    inner_y: int,
    eq_hist: bool = False,  # RGB 通常不需要均衡化
    blur_ksize: int = 0,
    prefer_sb: bool = True,  # 优先使用 SUBPIX 精度
    scale: float = 1.0,
    do_subpix: bool = True,
    subpix_win: int = 3,
    vis_dir: Optional[Path] = None,
):
    """批量处理 RGB 图像，自动检测角点"""
    images_dir = Path(images_dir)
    corners_dir = Path(corners_dir)
    
    # 查找所有图像文件
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_paths = []
    for ext in img_exts:
        image_paths.extend(sorted(images_dir.rglob(f"*{ext}")))
        image_paths.extend(sorted(images_dir.rglob(f"*{ext.upper()}")))
    
    if not image_paths:
        raise ValueError(f"在目录中未找到图像文件: {images_dir}")
    
    print(f"找到 {len(image_paths)} 张图像，开始检测角点...")
    print(f"棋盘规格: {inner_x} x {inner_y} 内角点\n")
    
    success_count = 0
    failed_list = []
    
    for img_path in image_paths:
        # 生成输出文件名：去掉扩展名，添加 _corners.npz
        base = img_path.stem
        output_path = corners_dir / f"{base}_corners.npz"
        
        # 如果已存在，跳过（可选：添加 --force 覆盖）
        if output_path.exists():
            print(f"⊘ 已存在，跳过: {base}")
            success_count += 1
            continue
        
        found, corners, raw = detect_corners(
            image_path=str(img_path),
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
            print(f"✗ 检测失败: {base} (找到 {corners.shape[0] if corners is not None else 0} 个角点，期望 {inner_x * inner_y})")
            failed_list.append(base)
            continue
        
        # 确保形状为 (N, 1, 2)
        if corners.ndim == 2:
            corners = corners[:, None, :]
        corners = corners.astype(np.float32)
        
        # 确保 0 号点在左上角（根据坐标判断）
        pts = corners.reshape(-1, 2)
        # 0号点应该是x+y最小的点，19号点应该是x+y最大的点
        sum_first = pts[0, 0] + pts[0, 1]
        sum_last = pts[-1, 0] + pts[-1, 1]
        if sum_first > sum_last:
            # 顺序反了，需要翻转
            corners = corners[::-1].copy()
            print(f"  [已调整] 角点顺序已自动翻转（确保0号在左上）")
        
        save_corners(corners, output_path)
        success_count += 1
        
        # 保存可视化
        if vis_dir:
            from detect_chessboard_corners import draw_vis
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis = draw_vis(raw, corners)
            vis_path = vis_dir / f"{base}_vis.png"
            cv2.imwrite(str(vis_path), vis)
            print(f"  可视化已保存: {vis_path}")
    
    print(f"\n{'='*50}")
    print(f"完成！成功: {success_count}/{len(image_paths)}")
    if failed_list:
        print(f"失败列表 ({len(failed_list)}):")
        for base in failed_list:
            print(f"  - {base}")
        print(f"\n提示：失败的图像可能需要调整检测参数或手动标注")


def main():
    parser = argparse.ArgumentParser(description="自动检测 RGB 图像棋盘角点并保存为 npz")
    parser.add_argument("--images-dir", required=True, type=str, help="RGB 图像目录（支持子目录递归）")
    parser.add_argument("--corners-dir", required=True, type=str, help="角点文件保存目录")
    parser.add_argument("--inner-corners-x", type=int, default=5, help="水平内角点个数（默认 6）")
    parser.add_argument("--inner-corners-y", type=int, default=4, help="垂直内角点个数（默认 5）")
    parser.add_argument("--eq-hist", action="store_true", help="启用直方图均衡化（RGB 通常不需要）")
    parser.add_argument("--blur", type=int, default=0, help="高斯模糊核大小（0=禁用）")
    parser.add_argument("--no-subpix", action="store_true", help="禁用亚像素优化")
    parser.add_argument("--subpix-win", type=int, default=3, help="亚像素窗口大小")
    parser.add_argument("--scale", type=float, default=1.0, help="图像缩放比例（默认 1.0）")
    parser.add_argument("--save-vis", type=str, default=None, help="保存角点可视化的目录")
    
    args = parser.parse_args()
    
    vis_dir = Path(args.save_vis) if args.save_vis else None
    
    process_images(
        images_dir=Path(args.images_dir),
        corners_dir=Path(args.corners_dir),
        inner_x=args.inner_corners_x,
        inner_y=args.inner_corners_y,
        eq_hist=args.eq_hist,
        blur_ksize=args.blur,
        prefer_sb=True,
        scale=args.scale,
        do_subpix=(not args.no_subpix),
        subpix_win=args.subpix_win,
        vis_dir=vis_dir,
    )


if __name__ == "__main__":
    main()

