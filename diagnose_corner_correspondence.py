#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断 RGB 与 ToF 角点对应关系，可视化角点编号与连线。
"""

import argparse
from pathlib import Path
import numpy as np
import cv2


def visualize_corner_order(img_path, corners, title, output_path, scale_factor=8.0):
    """可视化角点顺序：标注编号并连线"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] 无法读取图像: {img_path}")
        return
    
    h, w = img.shape[:2]
    
    # 如果图像太小，先放大
    if max(w, h) < 500:
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        corners_scaled = corners * scale_factor
        print(f"  图像已放大 {scale_factor}x: {w}x{h} → {img.shape[1]}x{img.shape[0]}")
    else:
        corners_scaled = corners.copy()
    
    pts = corners_scaled.reshape(-1, 2)
    
    # 根据图像尺寸自动调整参数
    img_size = max(img.shape[:2])
    line_thickness = max(1, int(img_size / 500))
    radius_normal = max(3, int(img_size / 200))
    radius_special = max(5, int(img_size / 100))
    font_scale = max(0.3, img_size / 1000)
    font_thickness = max(1, int(img_size / 500))
    text_offset = max(5, int(img_size / 200))
    
    # 画连线（按顺序）
    for i in range(len(pts) - 1):
        pt1 = tuple(pts[i].astype(int))
        pt2 = tuple(pts[i + 1].astype(int))
        cv2.line(img, pt1, pt2, (0, 255, 255), line_thickness)
    
    # 画角点与编号
    for i, pt in enumerate(pts):
        x, y = int(pt[0]), int(pt[1])
        # 编号颜色：0号红色，最后一号蓝色，其他绿色
        if i == 0:
            color = (0, 0, 255)  # 红色：起始点
            radius = radius_special
        elif i == len(pts) - 1:
            color = (255, 0, 0)  # 蓝色：结束点
            radius = radius_special
        else:
            color = (0, 255, 0)  # 绿色
            radius = radius_normal
        
        cv2.circle(img, (x, y), radius, color, -1)
        cv2.putText(img, str(i), (x + text_offset, y - text_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    # 添加标题
    title_font_scale = max(0.5, img_size / 800)
    title_thickness = max(1, int(img_size / 400))
    cv2.putText(img, title, (30, int(50 * title_font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 
               title_font_scale, (255, 255, 0), title_thickness)
    
    cv2.imwrite(str(output_path), img)
    print(f"已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="诊断角点对应关系")
    parser.add_argument("--rgb-image", required=True)
    parser.add_argument("--tof-image", required=True)
    parser.add_argument("--rgb-corners", required=True)
    parser.add_argument("--tof-corners", required=True)
    parser.add_argument("--output-dir", required=True)
    
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载角点
    rgb_data = np.load(args.rgb_corners)
    tof_data = np.load(args.tof_corners)
    
    rgb_corners = rgb_data["corners"] if "corners" in rgb_data else rgb_data[rgb_data.files[0]]
    tof_corners = tof_data["corners"] if "corners" in tof_data else tof_data[tof_data.files[0]]
    
    # 确保形状
    if rgb_corners.ndim == 2:
        rgb_corners = rgb_corners[:, None, :]
    if tof_corners.ndim == 2:
        tof_corners = tof_corners[:, None, :]
    
    # 可视化
    visualize_corner_order(
        args.rgb_image, rgb_corners, 
        f"RGB: {len(rgb_corners)} corners", 
        out_dir / "rgb_order.png"
    )
    
    visualize_corner_order(
        args.tof_image, tof_corners, 
        f"ToF: {len(tof_corners)} corners", 
        out_dir / "tof_order.png"
    )
    
    print(f"\n请对比两张图：")
    print(f"  RGB: {out_dir / 'rgb_order.png'}")
    print(f"  ToF: {out_dir / 'tof_order.png'}")
    print(f"\n关键检查：")
    print(f"  - 0号点（红色）应该在棋盘的同一位置（通常是左上角）")
    print(f"  - 最后一号点（蓝色）应该在棋盘的同一位置（通常是右下角）")
    print(f"  - 连线方向应该一致（都是行优先或都是列优先）")


if __name__ == "__main__":
    main()

