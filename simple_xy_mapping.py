#!/usr/bin/env python3
"""
简化方案：只做ToF到RGB的x,y像素映射，深度值直接复用
假设：两相机在同一平面，主要是横向平移
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


def compute_homography_for_plane(K_tof, K_rgb, R, t, depth_plane=500):
    """
    计算特定深度平面的单应性矩阵
    H = K_rgb @ (R - t*n.T/d) @ K_tof^-1
    其中n=[0,0,1]是平面法向量，d是平面距离
    """
    n = np.array([[0], [0], [1]])
    H = K_rgb @ (R - (t.reshape(3, 1) @ n.T) / depth_plane) @ np.linalg.inv(K_tof)
    return H


def map_tof_to_rgb_simple(depth_map, K_tof, K_rgb, R, t, rgb_shape, 
                          min_depth=200, max_depth=800):
    """
    简化映射：使用平均深度计算单应性矩阵
    """
    h, w = depth_map.shape
    
    # 过滤有效深度
    valid_depths = depth_map[(depth_map > min_depth) & (depth_map < max_depth)]
    if len(valid_depths) == 0:
        return None, None
    
    # 使用平均深度作为平面深度
    avg_depth = valid_depths.mean()
    print(f"  平均深度: {avg_depth:.1f} mm")
    
    # 计算单应性矩阵
    H = compute_homography_for_plane(K_tof, K_rgb, R, t, avg_depth)
    
    # 对所有ToF像素进行映射
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid_mask = (depth_map > min_depth) & (depth_map < max_depth)
    
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = depth_map[valid_mask]
    
    # 用单应性矩阵映射
    tof_pts_homo = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)  # (3, N)
    rgb_pts_homo = H @ tof_pts_homo  # (3, N)
    
    # 归一化
    rgb_x = rgb_pts_homo[0, :] / rgb_pts_homo[2, :]
    rgb_y = rgb_pts_homo[1, :] / rgb_pts_homo[2, :]
    
    rgb_pts = np.stack([rgb_x, rgb_y], axis=-1)
    
    # 过滤在图像内的点
    rgb_h, rgb_w = rgb_shape
    in_bounds = (
        (rgb_pts[:, 0] >= 0) & (rgb_pts[:, 0] < rgb_w) &
        (rgb_pts[:, 1] >= 0) & (rgb_pts[:, 1] < rgb_h)
    )
    
    return rgb_pts[in_bounds], z_valid[in_bounds]


def main():
    parser = argparse.ArgumentParser(description='简化xy映射测试')
    parser.add_argument('depth_csv', help='深度CSV')
    parser.add_argument('rgb_image', help='RGB图像')
    parser.add_argument('--output', default='simple_mapping.png', help='输出文件')
    parser.add_argument('--min-depth', type=float, default=200)
    parser.add_argument('--max-depth', type=float, default=500)
    
    args = parser.parse_args()
    
    # 读取
    depth = parse_tof_csv(args.depth_csv)
    rgb_img = cv2.imread(args.rgb_image)
    
    print("简化映射方案测试")
    print("="*70)
    
    # Padding和旋转
    if depth.shape == (100, 100):
        depth = np.vstack([depth, np.full((1, 100), 3400.0)])
    depth_rot = np.rot90(depth, k=1)
    
    # 加载标定
    with open('data/stereo_extrinsics.json') as f:
        calib = json.load(f)
    
    K_tof = np.array(calib['K_tof'])
    K_rgb = np.array(calib['K_rgb'])
    R = np.array(calib['R'])
    t = np.array(calib['t']).reshape(-1)
    
    print(f"深度范围: {args.min_depth}-{args.max_depth} mm (不缩放)")
    
    # 简化映射
    rgb_pts, depths = map_tof_to_rgb_simple(
        depth_rot, K_tof, K_rgb, R, t, rgb_img.shape[:2],
        args.min_depth, args.max_depth
    )
    
    if rgb_pts is None:
        print("无有效点")
        return
    
    print(f"  投影点数: {len(rgb_pts)}")
    print(f"  投影中心: ({rgb_pts[:, 0].mean():.0f}, {rgb_pts[:, 1].mean():.0f})")
    
    # 绘制
    rgb_vis = rgb_img.copy()
    d_min, d_max = depths.min(), depths.max()
    normalized = ((depths - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_JET)
    colors = colormap.reshape(-1, 3)
    
    for (x, y), color in zip(rgb_pts.astype(int), colors):
        cv2.circle(rgb_vis, (x, y), 3, color.tolist(), -1)
    
    cv2.putText(rgb_vis, 'Simple XY Mapping (no Z scale)', 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imwrite(args.output, rgb_vis)
    print(f"  ✓ 保存: {args.output}")
    print("\n这种方法假设所有深度点在同一平面，不考虑深度变化")


if __name__ == '__main__':
    main()

