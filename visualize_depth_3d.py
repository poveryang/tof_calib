#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D可视化深度图CSV，验证深度单位
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path


def parse_depth_csv(csv_path):
    """解析深度CSV"""
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    image_rows = []
    for line in lines[1:]:  # 跳过元数据
        if line.strip():
            row_data = [float(x) if x else 0 for x in line.split(',')]
            image_rows.append(row_data)
    
    return np.array(image_rows, dtype=np.float32)


def depth_to_pointcloud(depth_map, invalid_value=3400.0, scale=1.0):
    """
    将深度图转换为3D点云
    
    参数:
        depth_map: 深度图数组
        invalid_value: 无效深度值
        scale: 深度缩放因子
        
    返回:
        points: 3D点 (N, 3) - [x, y, z]
        depths: 深度值 (N,)
    """
    h, w = depth_map.shape
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 过滤无效深度
    valid_mask = (depth_map > 0) & (depth_map < invalid_value)
    
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = depth_map[valid_mask] * scale
    
    # 简化的3D重建：假设正交投影（不考虑相机内参）
    # x, y就是像素坐标，z是深度
    x = u_valid
    y = v_valid
    z = z_valid
    
    points = np.stack([x, y, z], axis=-1)
    
    return points, z_valid


def visualize_3d(points, depths, title="3D Depth Visualization", depth_scale=1.0):
    """
    3D可视化点云
    
    参数:
        points: 3D点 (N, 3)
        depths: 深度值用于着色
        title: 标题
        depth_scale: 深度缩放因子（用于显示）
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用深度值着色
    scatter = ax.scatter(
        points[:, 0],  # x (列)
        points[:, 1],  # y (行)
        points[:, 2],  # z (深度)
        c=depths,
        cmap='jet',
        s=2,
        alpha=0.6
    )
    
    ax.set_xlabel('X (像素列)')
    ax.set_ylabel('Y (像素行)')
    ax.set_zlabel(f'Z (深度 mm, scale={depth_scale}x)')
    ax.set_title(title)
    
    # 添加色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Depth (mm)')
    
    # 反转Y轴使其符合图像坐标系习惯
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig, ax


def analyze_region(depth_map, center_row, center_col, radius=5, scale=1.0):
    """
    分析指定区域的深度统计
    
    参数:
        depth_map: 深度图
        center_row, center_col: 中心像素坐标
        radius: 区域半径（像素）
        scale: 深度缩放因子
    """
    h, w = depth_map.shape
    
    r_min = max(0, center_row - radius)
    r_max = min(h, center_row + radius + 1)
    c_min = max(0, center_col - radius)
    c_max = min(w, center_col + radius + 1)
    
    region = depth_map[r_min:r_max, c_min:c_max]
    valid_region = region[(region > 0) & (region < 3400)]
    
    if len(valid_region) == 0:
        print(f"  区域[{center_row},{center_col}]±{radius}: 无有效深度")
        return None
    
    stats = {
        'center_pixel': (center_row, center_col),
        'center_value': depth_map[center_row, center_col],
        'region_min': valid_region.min(),
        'region_max': valid_region.max(),
        'region_mean': valid_region.mean(),
        'region_median': np.median(valid_region),
        'scale': scale
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='3D可视化深度CSV')
    parser.add_argument('depth_csv', type=str, help='深度CSV文件路径')
    parser.add_argument('--scale', type=float, default=1.0, help='深度缩放因子（默认1.0）')
    parser.add_argument('--check-center', action='store_true', help='检查中心区域深度')
    parser.add_argument('--check-point', type=int, nargs=2, metavar=('ROW', 'COL'), 
                        help='检查指定点的深度，格式: 行 列')
    parser.add_argument('--output', type=str, default=None, help='保存3D图像路径')
    parser.add_argument('--no-show', action='store_true', help='不显示交互窗口')
    
    args = parser.parse_args()
    
    # 读取深度数据
    depth_csv_path = Path(args.depth_csv)
    if not depth_csv_path.exists():
        print(f"错误: 文件不存在 {depth_csv_path}")
        return
    
    print("=" * 70)
    print("深度图3D可视化")
    print("=" * 70)
    
    depth = parse_depth_csv(depth_csv_path)
    print(f"\n深度图尺寸: {depth.shape}")
    print(f"深度缩放因子: {args.scale}x")
    
    # 统计信息
    valid_depths = depth[(depth > 0) & (depth < 3400)]
    if len(valid_depths) > 0:
        print(f"\n深度统计（原始CSV值）:")
        print(f"  最小值: {valid_depths.min():.1f}")
        print(f"  最大值: {valid_depths.max():.1f}")
        print(f"  平均值: {valid_depths.mean():.1f}")
        print(f"  中位数: {np.median(valid_depths):.1f}")
        print(f"  有效像素: {len(valid_depths)}/{depth.size}")
        
        print(f"\n深度统计（缩放后 ×{args.scale}）:")
        scaled = valid_depths * args.scale
        print(f"  最小值: {scaled.min():.1f} mm")
        print(f"  最大值: {scaled.max():.1f} mm")
        print(f"  平均值: {scaled.mean():.1f} mm")
        print(f"  中位数: {np.median(scaled):.1f} mm")
    
    # 检查中心区域
    if args.check_center:
        h, w = depth.shape
        center_r, center_c = h // 2, w // 2
        print(f"\n检查中心区域 [{center_r}, {center_c}]±5像素:")
        stats = analyze_region(depth, center_r, center_c, radius=5, scale=args.scale)
        if stats:
            print(f"  中心点值: {stats['center_value']:.1f} (CSV原始)")
            print(f"           {stats['center_value'] * args.scale:.1f} mm (缩放后)")
            print(f"  区域平均: {stats['region_mean']:.1f} (CSV原始)")
            print(f"           {stats['region_mean'] * args.scale:.1f} mm (缩放后)")
            print(f"  区域范围: {stats['region_min']:.1f} - {stats['region_max']:.1f} (CSV原始)")
            print(f"           {stats['region_min'] * args.scale:.1f} - {stats['region_max'] * args.scale:.1f} mm (缩放后)")
            
            print(f"\n验证单位:")
            print(f"  如果实际测量是300mm:")
            print(f"    - 若CSV单位=mm, 应显示≈300")
            print(f"    - 若CSV单位=0.5mm, 应显示≈600, 需×0.5得300mm")
            print(f"    - 若CSV单位=2mm, 应显示≈150, 需×2得300mm")
            print(f"  当前中心值: {stats['center_value']:.1f}")
            print(f"  建议缩放因子: {300 / stats['center_value']:.2f}x (使其=300mm)")
    
    # 检查指定点
    if args.check_point:
        r, c = args.check_point
        print(f"\n检查指定点 [{r}, {c}]±5像素:")
        stats = analyze_region(depth, r, c, radius=5, scale=args.scale)
        if stats:
            print(f"  点值: {stats['center_value']:.1f} (CSV原始)")
            print(f"       {stats['center_value'] * args.scale:.1f} mm (缩放后)")
            print(f"  区域平均: {stats['region_mean'] * args.scale:.1f} mm")
    
    # 生成3D点云
    print(f"\n生成3D点云...")
    points, depths_scaled = depth_to_pointcloud(depth, scale=args.scale)
    print(f"  有效点数: {len(points)}")
    
    # 3D可视化
    print(f"\n创建3D可视化...")
    fig, ax = visualize_3d(points, depths_scaled, 
                          title=f"Depth 3D View (scale={args.scale}x)",
                          depth_scale=args.scale)
    
    # 保存或显示
    if args.output:
        output_path = Path(args.output)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 3D图像已保存: {output_path}")
    
    if not args.no_show:
        print(f"\n显示3D交互窗口（可旋转查看）...")
        plt.show()
    else:
        plt.close()
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()

