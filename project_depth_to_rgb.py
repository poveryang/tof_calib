#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToF深度映射到RGB图像 (使用单应性矩阵方法)

流程：
1. 读取IR和depth的CSV并验证对齐
2. 深度图逆时针旋转90度（与标定时ToF坐标系对齐）
3. 用单应性矩阵将ToF像素映射到RGB像素
4. 深度值(z)不参与坐标转换，直接复用原始mm值
5. 用颜色表征深度

说明：
- 这种方法适用于两相机基本平行、在同一平面的情况
- 深度值保持原始CSV值（单位mm），不需要缩放
"""

import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt


def parse_tof_csv(csv_path):
    """解析ToF CSV文件"""
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    image_rows = []
    for line in lines[1:]:  # 跳过元数据
        if line.strip():
            row_data = [float(x) if x else 0 for x in line.split(',')]
            image_rows.append(row_data)
    
    return np.array(image_rows, dtype=np.float32)


def compute_tof_to_rgb_homography():
    """
    从标定角点计算ToF到RGB的单应性矩阵
    这种方法直接做2D到2D映射，深度值不参与坐标转换
    """
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
    
    if len(tof_pts) == 0:
        raise ValueError("未找到标定角点数据")
    
    tof_pts = np.vstack(tof_pts).astype(np.float32)
    rgb_pts = np.vstack(rgb_pts).astype(np.float32)
    
    # 计算单应性矩阵
    H, mask = cv2.findHomography(tof_pts, rgb_pts, cv2.RANSAC, 5.0)
    
    return H


def project_depth_to_rgb(depth_map, homography_matrix, rgb_shape, 
                         min_depth=200, max_depth=600):
    """
    使用单应性矩阵将ToF深度图映射到RGB图像
    
    方法说明：
    - 直接用单应性矩阵做2D像素坐标映射（x,y）
    - 深度值(z)不参与坐标转换，直接复用原始值
    - 这适用于两相机基本平行、在同一平面的情况
    
    参数:
        depth_map: 深度图 (H x W)
        homography_matrix: 单应性矩阵 ToF→RGB
        rgb_shape: RGB图像尺寸 (height, width)
        min_depth: 最小有效深度 (mm)
        max_depth: 最大有效深度 (mm)
        
    返回:
        rgb_points: RGB图像上的像素坐标
        depths: 对应的深度值（原始值，未缩放）
    """
    h, w = depth_map.shape
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 过滤有效深度
    valid_mask = (depth_map >= min_depth) & (depth_map <= max_depth)
    
    u_valid = u[valid_mask].astype(np.float32)
    v_valid = v[valid_mask].astype(np.float32)
    z_valid = depth_map[valid_mask].astype(np.float32)
    
    if len(z_valid) == 0:
        return np.array([]), np.array([])
    
    # 使用单应性矩阵进行2D映射
    tof_pixels = np.stack([u_valid, v_valid], axis=-1).reshape(1, -1, 2)
    rgb_pixels = cv2.perspectiveTransform(tof_pixels, homography_matrix).reshape(-1, 2)
    
    # 过滤超出RGB图像范围的点
    rgb_h, rgb_w = rgb_shape
    in_bounds = (
        (rgb_pixels[:, 0] >= 0) & (rgb_pixels[:, 0] < rgb_w) &
        (rgb_pixels[:, 1] >= 0) & (rgb_pixels[:, 1] < rgb_h)
    )
    
    return rgb_pixels[in_bounds], z_valid[in_bounds]


def colorize_depth_value(depth_values):
    """
    将深度值转换为颜色
    使用jet色图: 近距离(蓝色) -> 中距离(绿色/黄色) -> 远距离(红色)
    """
    if len(depth_values) == 0:
        return np.array([])
    
    # 归一化深度值到0-255
    d_min, d_max = depth_values.min(), depth_values.max()
    if d_max > d_min:
        normalized = ((depth_values - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros(len(depth_values), dtype=np.uint8)
    
    # 应用jet色图
    colormap = cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_JET)
    colors = colormap.reshape(-1, 3)
    
    return colors


def main():
    # 设置路径
    data_dir = Path(__file__).parent / 'data' / 'test'
    
    print("=" * 70)
    print("ToF深度投影到RGB图像")
    print("=" * 70)
    
    # 1. 读取数据
    print("\n[1] 读取ToF数据...")
    depth = parse_tof_csv(data_dir / 'Depth_0_0.csv')
    ir = parse_tof_csv(data_dir / 'IR_0_0.csv')
    rgb_img = cv2.imread(str(data_dir / 'rgb_test.png'))
    
    print(f"    深度CSV尺寸: {depth.shape}")
    print(f"    IR CSV尺寸: {ir.shape}")
    
    # 标定时的ToF图像是101×100，需要将100×100的深度图padding到101×100
    if depth.shape == (100, 100):
        print(f"    ⚠ 深度图尺寸(100×100)与标定时ToF图像(101×100)不匹配，进行padding...")
        # 在底部添加一行，使其变为101×100
        depth = np.vstack([depth, np.full((1, depth.shape[1]), 3400.0)])  # 用无效值填充
        ir = np.vstack([ir, np.zeros((1, ir.shape[1]))])
        print(f"    ✓ Padding后深度图尺寸: {depth.shape}")
    
    print(f"    RGB图尺寸: {rgb_img.shape}")
    
    # 2. 验证IR和深度对齐
    print("\n[2] 验证IR和深度对齐...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # IR图
    axes[0].imshow(ir, cmap='gray')
    axes[0].set_title('IR Image')
    axes[0].axis('off')
    
    # 深度图
    depth_vis = depth.copy()
    depth_vis[depth_vis >= 3400] = np.nan
    im1 = axes[1].imshow(depth_vis, cmap='jet')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], label='Depth (mm)')
    
    # IR-深度叠加
    ir_norm = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ir_rgb = cv2.cvtColor(ir_norm, cv2.COLOR_GRAY2BGR)
    
    depth_clean = depth.copy()
    depth_clean[depth_clean >= 3400] = 0
    depth_norm = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    depth_colored[depth >= 3400] = 0
    
    overlay = cv2.addWeighted(ir_rgb, 0.3, depth_colored, 0.7, 0)
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title('IR-Depth Overlay (Alignment Check)')
    axes[2].axis('off')
    
    plt.tight_layout()
    alignment_path = data_dir / 'alignment_check.png'
    plt.savefig(alignment_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ 对齐检查图已保存: {alignment_path}")
    plt.close()
    
    # 3. 深度图逆时针旋转90度
    print("\n[3] 深度图逆时针旋转90度...")
    depth_rotated = np.rot90(depth, k=1)  # k=1逆时针
    ir_rotated = np.rot90(ir, k=1)
    print(f"    旋转后深度图尺寸: {depth_rotated.shape}")
    
    # 调试：检查几个关键位置
    print(f"\n[调试] 深度值检查:")
    print(f"  旋转后 中心(50,50): {depth_rotated[50, 50]:.1f} mm")
    print(f"  旋转后 左上(10,10): {depth_rotated[10, 10]:.1f} mm")
    print(f"  旋转后 右下(90,90): {depth_rotated[90, 90]:.1f} mm")
    
    # 可视化旋转，并标注区域
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(ir, cmap='gray')
    axes[0, 0].set_title('IR (Original 101x100)')
    axes[0, 0].plot(10, 10, 'ro', markersize=10, label='Top-Left')
    axes[0, 0].plot(50, 50, 'go', markersize=10, label='Center')
    axes[0, 0].plot(90, 90, 'bo', markersize=10, label='Bottom-Right')
    axes[0, 0].legend()
    
    depth_vis_orig = depth.copy()
    depth_vis_orig[depth_vis_orig >= 3400] = np.nan
    axes[0, 1].imshow(depth_vis_orig, cmap='jet')
    axes[0, 1].set_title('Depth (Original)')
    axes[0, 1].plot(10, 10, 'ro', markersize=10)
    axes[0, 1].plot(50, 50, 'go', markersize=10)
    axes[0, 1].plot(90, 90, 'bo', markersize=10)
    
    axes[1, 0].imshow(ir_rotated, cmap='gray')
    axes[1, 0].set_title('IR (Rotated 100x101)')
    axes[1, 0].plot(40, 30, 'ro', markersize=10, label='From[10,10]?')
    axes[1, 0].plot(50, 50, 'go', markersize=10, label='Center')
    axes[1, 0].plot(60, 70, 'bo', markersize=10, label='From[90,90]?')
    axes[1, 0].legend()
    
    depth_vis_rot = depth_rotated.copy()
    depth_vis_rot[depth_vis_rot >= 3400] = np.nan
    axes[1, 1].imshow(depth_vis_rot, cmap='jet')
    axes[1, 1].set_title('Depth (Rotated)')
    axes[1, 1].plot(40, 30, 'ro', markersize=10)
    axes[1, 1].plot(50, 50, 'go', markersize=10)
    axes[1, 1].plot(60, 70, 'bo', markersize=10)
    
    plt.tight_layout()
    rotation_path = data_dir / 'rotation_check.png'
    plt.savefig(rotation_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ 旋转检查图已保存: {rotation_path}")
    plt.close()
    
    # 4. 计算ToF到RGB的单应性矩阵
    print("\n[4] 计算单应性矩阵...")
    print("    使用标定时的角点对应关系计算单应性矩阵")
    print("    这种方法直接做2D像素映射，深度值不参与坐标转换")
    
    H = compute_tof_to_rgb_homography()
    print(f"    ✓ 单应性矩阵已计算")
    
    # 5. 投影深度到RGB图像
    print("\n[5] 映射深度点到RGB图像...")
    
    # 设置深度有效范围（CSV原始值，单位mm，不需要缩放）
    MIN_DEPTH = 200  # mm
    MAX_DEPTH = 800  # mm
    
    rgb_points, depths = project_depth_to_rgb(
        depth_rotated,
        H,
        rgb_img.shape[:2],
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH
    )
    
    print(f"    深度范围: {MIN_DEPTH} - {MAX_DEPTH} mm (原始CSV值)")
    print(f"    映射到RGB的点: {len(rgb_points)}")
    if len(depths) > 0:
        print(f"    深度值范围: {depths.min():.1f} - {depths.max():.1f} mm")
        print(f"    投影中心: ({rgb_points[:, 0].mean():.0f}, {rgb_points[:, 1].mean():.0f})")
    
    # 6. 在RGB图像上绘制深度点
    print("\n[6] 生成深度映射可视化...")
    
    # 创建投影图像
    rgb_with_depth = rgb_img.copy()
    
    if len(depths) > 0:
        # 获取深度对应的颜色
        colors = colorize_depth_value(depths)
        
        # 绘制每个深度点（使用圆点）
        for (x, y), color in zip(rgb_points.astype(int), colors):
            cv2.circle(rgb_with_depth, (x, y), 3, color.tolist(), -1)
        
        print(f"    ✓ 已绘制 {len(depths)} 个深度点")
    else:
        print(f"    ⚠ 在指定范围内没有有效深度点")
    
    # 添加深度色条说明
    # 创建色条
    colorbar_height = 300
    colorbar_width = 30
    colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
    
    depth_range = np.linspace(0, 255, colorbar_height).astype(np.uint8)
    for i in range(colorbar_height):
        color = cv2.applyColorMap(np.array([[depth_range[i]]], dtype=np.uint8), 
                                   cv2.COLORMAP_JET)[0, 0]
        colorbar[i, :] = color
    
    # 在图像右侧添加色条
    h, w = rgb_with_depth.shape[:2]
    margin = 50
    
    # 扩展图像以容纳色条
    rgb_extended = np.ones((h, w + colorbar_width + margin * 2, 3), dtype=np.uint8) * 255
    rgb_extended[:h, :w] = rgb_with_depth
    
    # 添加色条
    y_start = (h - colorbar_height) // 2
    rgb_extended[y_start:y_start+colorbar_height, w+margin:w+margin+colorbar_width] = colorbar
    
    # 添加文字标注
    if len(depths) > 0:
        d_min, d_max = depths.min(), depths.max()
        cv2.putText(rgb_extended, f'{d_max:.0f}mm', 
                    (w+margin+colorbar_width+5, y_start+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(rgb_extended, f'{d_min:.0f}mm', 
                    (w+margin+colorbar_width+5, y_start+colorbar_height-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(rgb_extended, 'Depth', 
                    (w+margin+colorbar_width+5, y_start+colorbar_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        cv2.putText(rgb_extended, 'No data', 
                    (w+margin, y_start+colorbar_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 保存结果
    output_path = data_dir / 'depth_projected_on_rgb.png'
    cv2.imwrite(str(output_path), rgb_extended)
    print(f"    ✓ 深度投影图已保存: {output_path}")
    
    # 7. 创建对比视图
    print("\n[7] 创建对比视图...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 原始RGB图像
    axes[0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original RGB Image')
    axes[0].axis('off')
    
    # RGB + 深度投影
    axes[1].imshow(cv2.cvtColor(rgb_extended, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'RGB with Depth Projection ({len(rgb_points)} points)')
    axes[1].axis('off')
    
    plt.tight_layout()
    comparison_path = data_dir / 'projection_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ 对比图已保存: {comparison_path}")
    plt.close()
    
    # 总结
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print(f"\n生成的文件:")
    print(f"  1. {data_dir}/alignment_check.png         - IR与深度对齐验证")
    print(f"  2. {data_dir}/rotation_check.png          - 旋转前后对比")
    print(f"  3. {data_dir}/depth_projected_on_rgb.png  - 深度投影到RGB (主要结果)")
    print(f"  4. {data_dir}/projection_comparison.png   - 投影前后对比")
    
    print(f"\n映射统计:")
    total_valid = np.sum((depth_rotated >= MIN_DEPTH) & (depth_rotated <= MAX_DEPTH))
    print(f"  - 深度范围: {MIN_DEPTH}-{MAX_DEPTH} mm")
    print(f"  - 范围内深度点: {total_valid}")
    print(f"  - 成功映射到RGB: {len(rgb_points)}")
    if total_valid > 0:
        print(f"  - 映射率: {len(rgb_points) / total_valid * 100:.1f}%")
    print(f"\n说明:")
    print(f"  - 使用单应性矩阵进行2D像素映射")
    print(f"  - 深度值(z)不参与坐标转换，保持原始mm值")
    print(f"  - 适用于两相机基本平行、在同一平面的情况")


if __name__ == '__main__':
    main()

