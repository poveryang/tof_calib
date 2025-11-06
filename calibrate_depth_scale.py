#!/usr/bin/env python3
"""
通过已知真实距离的点，反推正确的深度缩放函数
"""

import numpy as np
import cv2
import json
from pathlib import Path
import argparse


def parse_tof_csv(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    image_rows = []
    for line in lines[1:]:
        if line.strip():
            row_data = [float(x) if x else 0 for x in line.split(',')]
            image_rows.append(row_data)
    return np.array(image_rows, dtype=np.float32)


def find_best_scale(tof_pixel, depth_csv_value, rgb_pixel_target, 
                   K_tof, dist_tof, K_rgb, dist_rgb, R, t):
    """
    通过一个已知对应关系的点，反推最佳缩放因子
    
    参数:
        tof_pixel: ToF像素坐标 (u, v)
        depth_csv_value: CSV中的深度值
        rgb_pixel_target: 应该投影到的RGB像素位置
        ...标定参数...
    """
    # 测试不同缩放因子
    best_scale = None
    best_error = float('inf')
    
    print(f"\n测试ToF像素{tof_pixel}，CSV深度={depth_csv_value:.0f}，目标RGB={rgb_pixel_target}")
    print("="*60)
    
    for scale in np.arange(0.5, 4.0, 0.1):
        depth_mm = depth_csv_value * scale
        
        # 反投影
        tof_px = np.array([[[tof_pixel[0], tof_pixel[1]]]], dtype=np.float32)
        norm_pt = cv2.undistortPoints(tof_px, K_tof, dist_tof).reshape(2)
        
        X_tof = np.array([norm_pt[0] * depth_mm, norm_pt[1] * depth_mm, depth_mm])
        X_rgb = R @ X_tof + t
        
        if X_rgb[2] < 50:
            continue
        
        # 投影
        X_rgb_proj = X_rgb.reshape(1, 1, 3).astype(np.float32)
        proj, _ = cv2.projectPoints(
            X_rgb_proj, np.zeros((3, 1)), np.zeros((3, 1)), K_rgb, dist_rgb
        )
        proj_xy = proj[0, 0]
        
        # 计算误差
        error = np.linalg.norm(proj_xy - rgb_pixel_target)
        
        if error < best_error:
            best_error = error
            best_scale = scale
        
        if scale in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            print(f"Scale {scale:.1f}x: 深度={depth_mm:6.0f}mm, 投影=({proj_xy[0]:6.0f},{proj_xy[1]:6.0f}), 误差={error:6.1f}px")
    
    print("="*60)
    print(f"最佳缩放因子: {best_scale:.2f}x (误差={best_error:.1f}px)")
    return best_scale


def main():
    parser = argparse.ArgumentParser(description='通过对应点反推深度缩放因子')
    parser.add_argument('depth_csv', help='深度CSV文件')
    parser.add_argument('--tof-point', type=int, nargs=2, required=True,
                       metavar=('ROW', 'COL'), help='ToF像素坐标(行,列)')
    parser.add_argument('--depth-value', type=float, help='该点的CSV深度值（不提供则自动读取）')
    parser.add_argument('--rgb-point', type=int, nargs=2, required=True,
                       metavar=('X', 'Y'), help='对应的RGB像素坐标(x,y)')
    
    args = parser.parse_args()
    
    # 读取深度
    depth = parse_tof_csv(args.depth_csv)
    
    # Padding和旋转
    if depth.shape == (100, 100):
        depth = np.vstack([depth, np.full((1, 100), 3400.0)])
    depth_rot = np.rot90(depth, k=1)  # 假设逆时针90度
    
    # 获取深度值
    tof_r, tof_c = args.tof_point
    if args.depth_value:
        depth_value = args.depth_value
    else:
        depth_value = depth_rot[tof_r, tof_c]
    
    print("输入数据:")
    print(f"  ToF像素(旋转后): [{tof_r}, {tof_c}]")
    print(f"  CSV深度值: {depth_value:.0f}")
    print(f"  目标RGB像素: {args.rgb_point}")
    
    # 加载标定
    with open('data/stereo_extrinsics.json') as f:
        calib = json.load(f)
    
    K_tof = np.array(calib['K_tof'])
    dist_tof = np.array(calib['dist_tof']).reshape(-1)
    K_rgb = np.array(calib['K_rgb'])
    dist_rgb = np.array(calib['dist_rgb']).reshape(-1)
    R = np.array(calib['R'])
    t = np.array(calib['t']).reshape(-1)
    
    # 反推最佳缩放
    best_scale = find_best_scale(
        (tof_c, tof_r),  # undistortPoints需要(u,v)即(列,行)
        depth_value,
        args.rgb_point,
        K_tof, dist_tof, K_rgb, dist_rgb, R, t
    )
    
    print(f"\n建议使用缩放因子: {best_scale:.2f}x")


if __name__ == '__main__':
    main()

