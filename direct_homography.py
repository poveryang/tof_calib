#!/usr/bin/env python3
"""
直接从ToF-RGB对应点计算单应性矩阵
不依赖外参，纯粹的2D到2D映射
"""

import numpy as np
import cv2
import argparse
from pathlib import Path


def parse_tof_csv(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    image_rows = []
    for line in lines[1:]:
        if line.strip():
            row_data = [float(x) if x else 0 for x in line.split(',')]
            image_rows.append(row_data)
    return np.array(image_rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='用对应点计算单应性矩阵')
    parser.add_argument('depth_csv', help='深度CSV')
    parser.add_argument('rgb_image', help='RGB图像')
    parser.add_argument('--use-calibration-points', action='store_true',
                       help='使用标定时的角点计算单应性')
    parser.add_argument('--output', default='homography_mapping.png')
    
    args = parser.parse_args()
    
    depth = parse_tof_csv(args.depth_csv)
    rgb_img = cv2.imread(args.rgb_image)
    
    # Padding和旋转
    if depth.shape == (100, 100):
        depth = np.vstack([depth, np.full((1, 100), 3400.0)])
    depth_rot = np.rot90(depth, k=1)
    
    print("使用单应性矩阵直接映射ToF→RGB")
    print("="*70)
    
    if args.use_calibration_points:
        # 使用标定时的角点计算单应性
        tof_pts = []
        rgb_pts = []
        
        for i in range(15):
            try:
                tof_c = np.load(f'data/tof_corners_rotated/{i}_corners.npz')['corners']
                rgb_c = np.load(f'data/rgb_corners/{i}_corners.npz')['corners']
                tof_pts.append(tof_c.reshape(-1, 2))
                rgb_pts.append(rgb_c.reshape(-1, 2))
            except:
                pass
        
        tof_pts = np.vstack(tof_pts).astype(np.float32)
        rgb_pts = np.vstack(rgb_pts).astype(np.float32)
        
        print(f"使用{len(tof_pts)}个标定角点")
        print(f"  ToF角点范围: X[{tof_pts[:, 0].min():.0f}, {tof_pts[:, 0].max():.0f}], Y[{tof_pts[:, 1].min():.0f}, {tof_pts[:, 1].max():.0f}]")
        print(f"  RGB角点范围: X[{rgb_pts[:, 0].min():.0f}, {rgb_pts[:, 0].max():.0f}], Y[{rgb_pts[:, 1].min():.0f}, {rgb_pts[:, 1].max():.0f}]")
        
        # 计算单应性矩阵
        H, mask = cv2.findHomography(tof_pts, rgb_pts, cv2.RANSAC, 5.0)
        
        print(f"\n单应性矩阵H (ToF→RGB):")
        print(H)
        
        # 用单应性映射所有ToF像素
        h, w = depth_rot.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = (depth_rot > 150) & (depth_rot < 400)
        
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = depth_rot[valid]
        
        tof_pixels = np.stack([u_valid, v_valid], axis=-1).astype(np.float32)
        rgb_pixels = cv2.perspectiveTransform(tof_pixels.reshape(1, -1, 2), H).reshape(-1, 2)
        
        # 过滤在图像内
        rgb_h, rgb_w = rgb_img.shape[:2]
        in_bounds = (
            (rgb_pixels[:, 0] >= 0) & (rgb_pixels[:, 0] < rgb_w) &
            (rgb_pixels[:, 1] >= 0) & (rgb_pixels[:, 1] < rgb_h)
        )
        
        rgb_pixels_in = rgb_pixels[in_bounds]
        z_in = z_valid[in_bounds]
        
        print(f"\n单应性映射结果:")
        print(f"  投影点数: {len(rgb_pixels_in)}")
        print(f"  投影中心: ({rgb_pixels_in[:, 0].mean():.0f}, {rgb_pixels_in[:, 1].mean():.0f})")
        
        # 绘制
        rgb_vis = rgb_img.copy()
        d_min, d_max = z_in.min(), z_in.max()
        normalized = ((z_in - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_JET)
        colors = colormap.reshape(-1, 3)
        
        for (x, y), color in zip(rgb_pixels_in.astype(int), colors):
            cv2.circle(rgb_vis, (x, y), 3, color.tolist(), -1)
        
        cv2.putText(rgb_vis, 'Direct Homography (no 3D transform)', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imwrite(args.output, rgb_vis)
        print(f"  ✓ 保存: {args.output}")
        
        print(f"\n这种方法:")
        print(f"  - 直接2D到2D映射，不经过3D空间")
        print(f"  - 深度值z完全不参与坐标转换")
        print(f"  - 只是用于着色显示")
        print(f"  - 如果这个结果正确，说明问题在于3D变换逻辑")


if __name__ == '__main__':
    main()

