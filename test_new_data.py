#!/usr/bin/env python3
"""
测试新采集的RGB和深度数据
用于验证投影配置和深度单位
"""

import numpy as np
import cv2
import json
from pathlib import Path
import argparse


def parse_tof_csv(csv_path):
    """解析ToF CSV"""
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    image_rows = []
    for line in lines[1:]:
        if line.strip():
            row_data = [float(x) if x else 0 for x in line.split(',')]
            image_rows.append(row_data)
    return np.array(image_rows, dtype=np.float32)


def project_depth(depth_rot, K_tof, dist_tof, K_rgb, dist_rgb, R, t, rgb_shape, 
                  depth_scale, min_depth, max_depth):
    """投影深度到RGB"""
    h, w = depth_rot.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    depth_scaled = depth_rot * depth_scale
    valid = (depth_scaled >= min_depth) & (depth_scaled <= max_depth)
    
    u_valid = u[valid].astype(np.float32)
    v_valid = v[valid].astype(np.float32)
    z_valid = depth_scaled[valid].astype(np.float32)
    
    if len(z_valid) == 0:
        return None, None
    
    # 投影
    tof_pixels = np.stack([u_valid, v_valid], axis=-1).reshape(-1, 1, 2)
    norm_pts = cv2.undistortPoints(tof_pixels, K_tof, dist_tof).reshape(-1, 2)
    
    X_tof = np.zeros((len(z_valid), 3), dtype=np.float32)
    X_tof[:, 0] = norm_pts[:, 0] * z_valid
    X_tof[:, 1] = norm_pts[:, 1] * z_valid
    X_tof[:, 2] = z_valid
    
    t_vec = t.reshape(3, 1)
    X_rgb = (R @ X_tof.T + t_vec).T
    
    valid_z = X_rgb[:, 2] > 50
    X_rgb = X_rgb[valid_z]
    z_valid = z_valid[valid_z]
    
    if len(X_rgb) == 0:
        return None, None
    
    X_rgb_proj = X_rgb.reshape(-1, 1, 3).astype(np.float32)
    proj_rgb, _ = cv2.projectPoints(
        X_rgb_proj, np.zeros((3, 1)), np.zeros((3, 1)), K_rgb, dist_rgb
    )
    proj_rgb = proj_rgb.reshape(-1, 2)
    
    # 过滤在图像内
    rgb_h, rgb_w = rgb_shape
    in_bounds = (
        (proj_rgb[:, 0] >= 0) & (proj_rgb[:, 0] < rgb_w) &
        (proj_rgb[:, 1] >= 0) & (proj_rgb[:, 1] < rgb_h)
    )
    
    return proj_rgb[in_bounds], z_valid[in_bounds]


def main():
    parser = argparse.ArgumentParser(description='测试新数据的投影')
    parser.add_argument('depth_csv', help='深度CSV文件')
    parser.add_argument('rgb_image', help='RGB图像文件')
    parser.add_argument('--output-dir', default='output', help='输出目录')
    parser.add_argument('--min-depth', type=float, default=200, help='最小深度(CSV值)')
    parser.add_argument('--max-depth', type=float, default=500, help='最大深度(CSV值)')
    
    args = parser.parse_args()
    
    # 读取数据
    depth = parse_tof_csv(args.depth_csv)
    rgb_img = cv2.imread(args.rgb_image)
    
    print("=" * 70)
    print("测试新采集的数据")
    print("=" * 70)
    print(f"\n深度CSV: {args.depth_csv}")
    print(f"RGB图像: {args.rgb_image}")
    print(f"深度图尺寸: {depth.shape}")
    print(f"RGB图尺寸: {rgb_img.shape}")
    
    # Padding
    if depth.shape == (100, 100):
        depth = np.vstack([depth, np.full((1, 100), 3400.0)])
        print(f"Padding后: {depth.shape}")
    
    # 加载标定
    with open('data/stereo_extrinsics.json') as f:
        calib = json.load(f)
    
    K_tof = np.array(calib['K_tof'])
    dist_tof = np.array(calib['dist_tof']).reshape(-1)
    K_rgb = np.array(calib['K_rgb'])
    dist_rgb = np.array(calib['dist_rgb']).reshape(-1)
    R = np.array(calib['R'])
    t = np.array(calib['t']).reshape(-1)
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 测试配置
    configs = [
        (0, 1.0, "不旋转_scale1.0"),
        (1, 1.0, "逆时针90_scale1.0"),
        (0, 2.0, "不旋转_scale2.0"),
        (1, 2.0, "逆时针90_scale2.0"),
        (1, 2.1, "逆时针90_scale2.1"),
        (1, 2.2, "逆时针90_scale2.2"),
        (1, 2.3, "逆时针90_scale2.3"),
        (1, 2.4, "逆时针90_scale2.4"),
        (1, 2.5, "逆时针90_scale2.5"),
        (1, 2.6, "逆时针90_scale2.6"),
        (1, 2.7, "逆时针90_scale2.7"),
        (1, 2.8, "逆时针90_scale2.8"),
        (1, 2.9, "逆时针90_scale2.9"),
    ]
    
    print(f"\n测试4种配置:")
    print("=" * 70)
    
    for rotation_k, scale, name in configs:
        # 旋转
        depth_rot = np.rot90(depth, k=rotation_k)
        
        # 投影
        min_d = args.min_depth * scale
        max_d = args.max_depth * scale
        
        proj_rgb, depths = project_depth(
            depth_rot, K_tof, dist_tof, K_rgb, dist_rgb, R, t,
            rgb_img.shape[:2], scale, min_d, max_d
        )
        
        if proj_rgb is None:
            print(f"\n{name}: 无有效点")
            continue
        
        print(f"\n{name}:")
        print(f"  旋转: k={rotation_k}")
        print(f"  缩放: {scale}x")
        print(f"  深度范围: {min_d:.0f}-{max_d:.0f}mm")
        print(f"  投影点数: {len(proj_rgb)}")
        print(f"  投影中心: ({proj_rgb[:, 0].mean():.0f}, {proj_rgb[:, 1].mean():.0f})")
        
        # 绘制
        rgb_vis = rgb_img.copy()
        
        d_min, d_max = depths.min(), depths.max()
        normalized = ((depths - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_JET)
        colors = colormap.reshape(-1, 3)
        
        for (x, y), color in zip(proj_rgb.astype(int), colors):
            cv2.circle(rgb_vis, (x, y), 3, color.tolist(), -1)
        
        # 添加标注
        cv2.putText(rgb_vis, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        filename = f'{name}.png'
        cv2.imwrite(str(output_dir / filename), rgb_vis)
        print(f"  ✓ 保存: {output_dir}/{filename}")
    
    print("\n" + "=" * 70)
    print("完成！请查看输出图像，看哪个配置正确")
    print("如果都不对，说明标定参数可能有问题，需要重新标定")
    print("=" * 70)


if __name__ == '__main__':
    main()

