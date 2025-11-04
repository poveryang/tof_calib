#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 ToF 的 IR 图上检测棋盘格内角点，并输出可视化结果。

默认参数：
- 内角点：5 x 4（对应 6 x 5 方格，与本仓库默认棋盘一致）
- 自动灰度、可选预处理（直方图均衡/高斯模糊），可选 ROI

使用示例：
  python3 detect_chessboard_corners.py \
    --image IR_0_0.png \
    --inner-corners-x 5 --inner-corners-y 4 \
    --save-vis out_corners.png

注意：对于低分辨率或低对比度的 ToF IR，建议开启均衡化与适度平滑；
若为 16-bit IR，会自动做稳健归一化到 8-bit。
"""

from __future__ import annotations

import argparse
from typing import Tuple

import cv2
import numpy as np


def parse_roi(roi_str: str | None) -> Tuple[int, int, int, int] | None:
    if not roi_str:
        return None
    try:
        parts = [int(p) for p in roi_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        x, y, w, h = parts
        if min(w, h) <= 0:
            raise ValueError
        return x, y, w, h
    except Exception:
        raise argparse.ArgumentTypeError("ROI 格式应为 x,y,w,h，且 w,h>0")


def to_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    # 稳健拉伸到 8-bit（1~99百分位），适配 ToF 16位动态范围
    lo, hi = np.percentile(gray.astype(np.float32), (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(gray.min()), float(gray.max())
        if hi <= lo:
            return np.zeros_like(gray, dtype=np.uint8)
    scaled = (gray.astype(np.float32) - lo) * (255.0 / max(1e-6, (hi - lo)))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def preprocess(gray: np.ndarray, eq_hist: bool, blur_ksize: int, invert: bool) -> np.ndarray:
    result = to_uint8(gray)
    if invert:
        result = cv2.bitwise_not(result)
    if eq_hist:
        # CLAHE 相比全局均衡化更稳健
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(result)
    if blur_ksize > 0:
        k = max(1, blur_ksize | 1)  # odd
        result = cv2.GaussianBlur(result, (k, k), 0)
    return result


def preprocess_with_intermediates(
    gray: np.ndarray, eq_hist: bool, blur_ksize: int, invert: bool
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (最终预处理结果, 均衡化后的图像)。均衡图在模糊之前保存，便于观察对比度提升。
    """
    base = to_uint8(gray)
    if invert:
        base = cv2.bitwise_not(base)
    if eq_hist:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_img = clahe.apply(base)
    else:
        eq_img = base.copy()
    result = eq_img
    if blur_ksize > 0:
        k = max(1, blur_ksize | 1)
        result = cv2.GaussianBlur(result, (k, k), 0)
    return result, eq_img


def detect_corners(
    image_path: str,
    inner_x: int,
    inner_y: int,
    eq_hist: bool,
    blur_ksize: int,
    invert: bool,
    scale: float,
    roi: Tuple[int, int, int, int] | None,
    use_fast_check: bool,
    prefer_sb: bool,
    debug: bool,
    do_subpix: bool,
    subpix_win: int,
):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # 已是单通道（典型 IR）
        gray = img

    if debug:
        print(f"原图信息: shape={gray.shape}, dtype={gray.dtype}, min={int(gray.min())}, max={int(gray.max())}")

    view = gray
    offset = (0, 0)
    if roi is not None:
        x, y, w, h = roi
        H, W = gray.shape[:2]
        x2 = max(0, min(W, x + w))
        y2 = max(0, min(H, y + h))
        x1 = max(0, min(W, x))
        y1 = max(0, min(H, y))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("ROI 超出范围或无效")
        view = gray[y1:y2, x1:x2]
        offset = (x1, y1)

    if debug:
        proc, eq_img = preprocess_with_intermediates(view, eq_hist=eq_hist, blur_ksize=blur_ksize, invert=invert)
        cv2.imwrite("_dbg_eq.png", to_uint8(eq_img))
        print("已保存: _dbg_eq.png (直方图均衡后，模糊前)")
        cv2.imwrite("_dbg_preproc.png", to_uint8(proc))
        print("已保存: _dbg_preproc.png (最终预处理结果)")
    else:
        proc, _ = preprocess_with_intermediates(view, eq_hist=eq_hist, blur_ksize=blur_ksize, invert=invert)

    # 可选：缩放后检测，低分辨率 ToF 可设 scale=2.0 提升鲁棒性
    resized = proc
    scale_used = 1.0
    if scale and abs(scale - 1.0) > 1e-3:
        scale_used = float(scale)
        new_w = max(1, int(proc.shape[1] * scale_used))
        new_h = max(1, int(proc.shape[0] * scale_used))
        resized = cv2.resize(proc, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        if debug:
            cv2.imwrite("_dbg_resized.png", to_uint8(resized))
            print(f"已保存: _dbg_resized.png (scale={scale_used})")

    flags = 0
    # 自适应阈值/快检可加速并提高鲁棒性
    flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
    flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
    if use_fast_check:
        flags |= cv2.CALIB_CB_FAST_CHECK

    pattern_size = (inner_x, inner_y)

    found = False
    corners = None

    # 优先使用更鲁棒的 SB 算法（OpenCV>=4.5），失败则回退
    if prefer_sb and hasattr(cv2, 'findChessboardCornersSB'):
        try:
            sb_ret = cv2.findChessboardCornersSB(resized, pattern_size, flags=0)
            # 兼容不同 OpenCV 绑定的返回：可能是 ndarray，或 (retval, corners)
            if isinstance(sb_ret, tuple):
                if len(sb_ret) == 2 and isinstance(sb_ret[0], (bool, np.bool_)):
                    found = bool(sb_ret[0])
                    corners = sb_ret[1]
                else:
                    corners = sb_ret[-1]
                    found = corners is not None
            else:
                corners = sb_ret
                found = corners is not None
        except Exception:
            found = False
            corners = None
    if not found:
        found, corners = cv2.findChessboardCorners(resized, patternSize=pattern_size, flags=flags)

    refined = None
    if corners is not None:
        # 若角点数量与期望不一致，则视为未找到
        expected = inner_x * inner_y
        c_arr = np.asarray(corners)
        c_count = c_arr.shape[0] if c_arr.ndim >= 2 else 0
        if c_count != expected:
            found = False
        else:
            found = True if found else True

    if found and corners is not None:
        # 标准化 corners 形状与类型为 (N,1,2) float32
        corners = np.asarray(corners)
        if corners.ndim == 2 and corners.shape[1] == 2:
            corners = corners.reshape((-1, 1, 2))
        corners = corners.astype(np.float32, copy=False)
        # 缩放回原尺寸坐标
        if abs(scale_used - 1.0) > 1e-3:
            corners = corners / scale_used
        # 亚像素细化（可选，窗口可调）
        if do_subpix:
            win = max(1, subpix_win | 1)  # odd
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
            refined = cv2.cornerSubPix(to_uint8(view), corners, (win, win), (-1, -1), term)
        else:
            refined = corners

        # 将 ROI 坐标移回全图
        refined[:, 0, 0] += offset[0]
        refined[:, 0, 1] += offset[1]

    return found, refined, img


def _color_wheel(n: int) -> list[tuple[int, int, int]]:
    if n <= 0:
        return []
    colors: list[tuple[int, int, int]] = []
    for i in range(n):
        hue = int(180.0 * i / max(1, n))  # OpenCV HSV hue range [0,180)
        hsv = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def draw_vis(
    img: np.ndarray,
    corners: np.ndarray | None,
) -> np.ndarray:
    # 统一转换为 3 通道 uint8，以便用纯红色绘制角点并正确保存
    if img.ndim == 3:
        # 若为彩色但非 uint8，则转灰度后稳健缩放到 8-bit，再转回 BGR
        if img.dtype != np.uint8:
            gray8 = to_uint8(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            vis = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()
    else:
        vis = cv2.cvtColor(to_uint8(img), cv2.COLOR_GRAY2BGR)
    if corners is not None:
        pts = np.asarray(corners)
        pts = pts.reshape(-1, 2) if pts.ndim >= 2 else np.empty((0, 2))
        # 固定半径为 1（原始分辨率）
        pr = 1
        # 画点
        int_pts = []
        for (x, y) in pts:
            cx, cy = int(round(float(x))), int(round(float(y)))
            int_pts.append((cx, cy))
            cv2.circle(vis, (cx, cy), pr, (0, 0, 255), -1)
        # 按序连接相邻角点
        for i in range(1, len(int_pts)):
            cv2.line(vis, int_pts[i-1], int_pts[i], (0, 255, 0), 1, cv2.LINE_AA)
        # 标注起点与终点（起点=蓝色，终点=黄色）
        if len(int_pts) > 0:
            sx, sy = int_pts[0]
            cv2.circle(vis, (sx, sy), pr, (255, 0, 0), -1)    # BGR: 蓝
        if len(int_pts) > 1:
            ex, ey = int_pts[-1]
            cv2.circle(vis, (ex, ey), pr, (0, 255, 255), -1)  # BGR: 黄
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="在 IR 图上检测棋盘内角点")
    parser.add_argument("--image", required=True, type=str, help="输入图像路径（如 IR_0_0.png）")
    parser.add_argument("--inner-corners-x", type=int, default=5, help="水平内角点个数（默认 5）")
    parser.add_argument("--inner-corners-y", type=int, default=4, help="垂直内角点个数（默认 4）")
    parser.add_argument("--eq-hist", default=True, action="store_true", help="启用 CLAHE 直方图均衡化")
    parser.add_argument("--blur", type=int, default=1, help="高斯模糊核大小（奇数，0 表示关闭；推荐 0 或 3）")
    parser.add_argument("--invert", action="store_true", help="反相（当棋盘亮/背景暗不利于检测时可尝试）")
    parser.add_argument("--scale", type=float, default=1.0, help="检测前放大比例，低分辨率可设 2.0")
    parser.add_argument("--roi", type=str, default=None, help="可选 ROI，格式 x,y,w,h（像素）")
    parser.add_argument("--fast-check", action="store_true", help="启用快速检查（更快但可能漏检）")
    parser.add_argument("--sb", action="store_true", help="优先使用 findChessboardCornersSB（更鲁棒，OpenCV>=4.5）")
    parser.add_argument("--debug", action="store_true", help="输出调试信息并保存中间结果")
    parser.add_argument("--save-vis", type=str, default=None, help="可视化结果输出路径（不填则不保存可视化）")
    parser.add_argument("--no-subpix", action="store_true", help="关闭亚像素细化（用于定位细化是否导致偏移）")
    parser.add_argument("--subpix-win", type=int, default=3, help="亚像素细化窗口大小（奇数像素，默认3）")
    parser.add_argument("--vis-scale", type=float, default=4.0, help="可视化输出放大倍数（默认4.0）")
    parser.add_argument("--save-corners", type=str, default=None, help="保存角点坐标到文件（.npz 格式，供标定使用）")

    args = parser.parse_args()

    roi = parse_roi(args.roi)
    found, corners, raw = detect_corners(
        image_path=args.image,
        inner_x=args.inner_corners_x,
        inner_y=args.inner_corners_y,
        eq_hist=args.eq_hist,
        blur_ksize=args.blur,
        invert=args.invert,
        scale=args.scale,
        roi=roi,
        use_fast_check=args.fast_check,
        prefer_sb=args.sb,
        debug=args.debug,
        do_subpix=not args.no_subpix,
        subpix_win=args.subpix_win,
    )

    if found and corners is not None:
        print(f"找到角点：{args.inner_corners_x} x {args.inner_corners_y}，共 {args.inner_corners_x*args.inner_corners_y} 个")
        # 保存角点坐标（供标定使用）
        if args.save_corners:
            np.savez_compressed(args.save_corners, corners=corners)
            print(f"角点已保存：{args.save_corners}")
        if args.save_vis:
            vis = draw_vis(raw, corners)
            if args.vis_scale and abs(args.vis_scale - 1.0) > 1e-3:
                h, w = vis.shape[:2]
                new_w = max(1, int(round(w * args.vis_scale)))
                new_h = max(1, int(round(h * args.vis_scale)))
                vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(args.save_vis, vis)
            print(f"可视化已保存：{args.save_vis}")
    else:
        if args.save_vis:
            # 即使未匹配到完整角点，也输出散点便于人工判断
            vis = draw_vis(raw, corners)
            if args.vis_scale and abs(args.vis_scale - 1.0) > 1e-3:
                h, w = vis.shape[:2]
                new_w = max(1, int(round(w * args.vis_scale)))
                new_h = max(1, int(round(h * args.vis_scale)))
                vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(args.save_vis, vis)
            print(f"未找到完整角点，已输出散点可视化：{args.save_vis}")


if __name__ == "__main__":
    main()


