#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于已提取的棋盘角点文件（*.npz）进行 ToF(以IR几何) 与 RGB 的外参标定与评估/可视化。

输入：
- 两路角点目录：每个样本一个 `<base>_corners.npz`（键名 `corners` 或第一个数组），形状 (N,1,2)
- 两路原图目录：用于读取图像尺寸与叠加可视化
- 可选两路内参 JSON：键名需兼容 {"camera_matrix": ..., "distortion_coefficients": ...}

输出：
- 立体外参与最终两路内参到 JSON
- 可选：叠加图（ToF→RGB 投影角点）与误差统计 CSV
- 可选：rectify 预览图
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from calibrate_mono import build_object_points


# ----------------------------- I/O 与工具函数 -----------------------------

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_image_by_base(images_dir: Path, base: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = images_dir / f"{base}{ext}"
        if p.exists():
            return p
    # 兼容大小写后缀
    for ext in tuple(e.upper() for e in IMG_EXTS):
        p = images_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None


def first_image_in_dir(images_dir: Path) -> Optional[Path]:
    for ext in IMG_EXTS:
        for p in sorted(images_dir.glob(f"*{ext}")):
            return p
    for ext in tuple(e.upper() for e in IMG_EXTS):
        for p in sorted(images_dir.glob(f"*{ext}")):
            return p
    return None


def load_corners_npz(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.load(str(path), allow_pickle=True)
    except Exception as e:
        print(f"[WARN] 加载角点失败: {path.name}: {e}")
        return None
    # 兼容键名或第一个数组
    corners = None
    if isinstance(data, np.lib.npyio.NpzFile):
        if "corners" in data.files:
            corners = data["corners"]
        else:
            if len(data.files) > 0:
                corners = data[data.files[0]]
    else:
        corners = np.array(data)
    if corners is None:
        print(f"[WARN] npz 中未找到角点数组: {path.name}")
        return None
    corners = np.asarray(corners)
    if corners.ndim == 2 and corners.shape[1] == 2:
        corners = corners[:, None, :]
    if corners.ndim != 3 or corners.shape[1:] != (1, 2):
        print(f"[WARN] 角点形状异常: {path.name}, got {corners.shape}, 期望 (N,1,2)")
        return None
    return corners.astype(np.float32)


def scan_bases(corners_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for p in sorted(corners_dir.glob("*_corners.npz")):
        base = p.name.replace("_corners.npz", "")
        if base:
            mapping[base] = p
    return mapping


def load_intrinsics_json(json_path: Optional[Path]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not json_path:
        return None
    if not json_path.exists():
        print(f"[WARN] 内参 JSON 不存在: {json_path}")
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        K = np.asarray(data.get("camera_matrix"), dtype=np.float64)
        dist = np.asarray(data.get("distortion_coefficients"), dtype=np.float64).reshape(-1, 1)
        if K.shape != (3, 3) or dist.ndim != 2 or dist.shape[1] != 1:
            raise ValueError("形状不匹配")
        return K, dist
    except Exception as e:
        print(f"[WARN] 读取内参 JSON 失败: {json_path}: {e}")
        return None


def ensure_image_size(images_dir: Path, base: Optional[str]) -> Tuple[int, int]:
    img_path = None
    if base:
        img_path = find_image_by_base(images_dir, base)
    if img_path is None:
        img_path = first_image_in_dir(images_dir)
    if img_path is None:
        raise FileNotFoundError(f"无法在目录中找到任意图像: {images_dir}")
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"读取图像失败: {img_path}")
    h, w = img.shape[:2]
    return int(w), int(h)


# ----------------------------- 数据结构 -----------------------------

@dataclass
class Sample:
    base: str
    rgb_corners: np.ndarray  # (N,1,2)
    tof_corners: np.ndarray  # (N,1,2)
    rgb_image_path: Optional[Path]
    tof_image_path: Optional[Path]


# ----------------------------- 内参估计 -----------------------------

def estimate_intrinsics_from_corners(
    samples: List[Sample],
    inner_x: int,
    inner_y: int,
    square_mm: float,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    objp = build_object_points(inner_x, inner_y, square_mm).astype(np.float32)
    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    for s in samples:
        if s.rgb_corners is None:
            continue
        if s.rgb_corners.shape[0] != inner_x * inner_y:
            continue
        objpoints.append(objp)
        imgpoints.append(s.rgb_corners.astype(np.float32))
    if len(objpoints) < 3:
        raise ValueError("用于内参估计的样本不足（需要≥3）")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return K, dist


def estimate_intrinsics_from_corners_tof(
    samples: List[Sample],
    inner_x: int,
    inner_y: int,
    square_mm: float,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    objp = build_object_points(inner_x, inner_y, square_mm).astype(np.float32)
    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    for s in samples:
        if s.tof_corners is None:
            continue
        if s.tof_corners.shape[0] != inner_x * inner_y:
            continue
        objpoints.append(objp)
        imgpoints.append(s.tof_corners.astype(np.float32))
    if len(objpoints) < 3:
        raise ValueError("用于ToF内参估计的样本不足（需要≥3）")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return K, dist


# ----------------------------- 立体标定 -----------------------------

def stereo_calibrate(
    samples: List[Sample],
    inner_x: int,
    inner_y: int,
    square_mm: float,
    K_rgb: np.ndarray,
    dist_rgb: np.ndarray,
    K_tof: np.ndarray,
    dist_tof: np.ndarray,
    image_size_rgb: Tuple[int, int],
    fix_intrinsic: bool,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    objp = build_object_points(inner_x, inner_y, square_mm).astype(np.float32)
    objpoints: List[np.ndarray] = []
    imgpoints_rgb: List[np.ndarray] = []
    imgpoints_tof: List[np.ndarray] = []

    for s in samples:
        if s.rgb_corners.shape[0] != inner_x * inner_y or s.tof_corners.shape[0] != inner_x * inner_y:
            continue
        objpoints.append(objp)
        imgpoints_rgb.append(s.rgb_corners.astype(np.float32))
        imgpoints_tof.append(s.tof_corners.astype(np.float32))

    if len(objpoints) < 3:
        raise ValueError("有效样本不足（需要≥3）")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    flags = 0
    if fix_intrinsic:
        flags |= cv2.CALIB_FIX_INTRINSIC
    else:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # 仅启用5参数畸变：k1,k2,p1,p2,k3
        flags |= cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
        if hasattr(cv2, "CALIB_FIX_S1_S2_S3_S4"):
            flags |= cv2.CALIB_FIX_S1_S2_S3_S4
        if hasattr(cv2, "CALIB_FIX_TAUX_TAUY"):
            flags |= cv2.CALIB_FIX_TAUX_TAUY

    # 注意：stereoCalibrate的参数顺序
    # R, T 是从第二个相机(imgpoints2)到第一个相机(imgpoints1)的变换
    # 我们传入的是 imgpoints_tof, imgpoints_rgb (ToF在前，RGB在后)
    # 所以返回的R,T是 RGB→ToF
    # 需要交换参数顺序，使返回的是 ToF→RGB
    ret, K_tof_out, dist_tof_out, K_rgb_out, dist_rgb_out, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_tof,      # 第一个相机：ToF
        imgpoints_rgb,      # 第二个相机：RGB
        K_tof.copy(),       # ToF内参
        dist_tof.copy(),
        K_rgb.copy(),       # RGB内参
        dist_rgb.copy(),
        image_size_rgb,
        criteria=criteria,
        flags=flags,
    )

    print(f"立体标定误差 (RMS): {ret:.4f} 像素")
    # 现在R,T是 ToF→RGB（从第一个相机ToF到第二个相机RGB）
    
    return (K_rgb_out, dist_rgb_out, R, T), (K_tof_out, dist_tof_out, E, F)


# ----------------------------- 误差评估 -----------------------------

def mean_reprojection_error_via_pnp(
    samples: List[Sample],
    inner_x: int,
    inner_y: int,
    square_mm: float,
    K_rgb: np.ndarray,
    dist_rgb: np.ndarray,
    K_tof: np.ndarray,
    dist_tof: np.ndarray,
    R_ir_to_rgb: np.ndarray,
    t_ir_to_rgb: np.ndarray,
) -> Tuple[float, List[Tuple[str, float]], Dict[str, np.ndarray]]:
    objp = build_object_points(inner_x, inner_y, square_mm).astype(np.float32)
    per_image_errors: List[Tuple[str, float]] = []
    projections: Dict[str, np.ndarray] = {}

    # 选择更稳健的PnP方案
    pnp_flags = None
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        pnp_flags = cv2.SOLVEPNP_IPPE
    elif hasattr(cv2, "SOLVEPNP_ITERATIVE"):
        pnp_flags = cv2.SOLVEPNP_ITERATIVE
    else:
        pnp_flags = 0

    all_errors: List[float] = []

    for s in samples:
        if s.tof_corners.shape[0] != objp.shape[0] or s.rgb_corners.shape[0] != objp.shape[0]:
            continue
        ok, rvec_ir, tvec_ir = cv2.solvePnP(
            objp,
            s.tof_corners,
            K_tof,
            dist_tof,
            flags=pnp_flags,
        )
        if not ok:
            print(f"[WARN] solvePnP 失败: {s.base}")
            continue
        R_ir, _ = cv2.Rodrigues(rvec_ir)
        # 将棋盘点从棋盘坐标到 ToF 相机坐标，再到 RGB 相机坐标
        # X_ir = R_ir * X_b + t_ir;  X_rgb = R * X_ir + t
        # objp 是 (N, 3)，转置成 (3, N) 便于矩阵运算
        Xb_T = objp.T  # (3, N)
        tvec_ir = tvec_ir.reshape(3, 1)  # 确保是 (3, 1)
        t_ir_to_rgb = t_ir_to_rgb.reshape(3, 1)  # 确保是 (3, 1)
        X_ir = R_ir @ Xb_T + tvec_ir  # (3, N)
        X_rgb = R_ir_to_rgb @ X_ir + t_ir_to_rgb  # (3, N)
        X_rgb = X_rgb.T.reshape(-1, 1, 3)  # (N, 1, 3)
        proj, _ = cv2.projectPoints(X_rgb, np.zeros((3, 1)), np.zeros((3, 1)), K_rgb, dist_rgb)
        proj = proj.astype(np.float32)  # 确保类型一致
        projections[s.base] = proj
        
        # 计算逐点误差用于调试
        rgb_pts = s.rgb_corners.reshape(-1, 2)
        proj_pts = proj.reshape(-1, 2)
        point_errors = np.linalg.norm(rgb_pts - proj_pts, axis=1)
        max_err_idx = np.argmax(point_errors)
        err = cv2.norm(s.rgb_corners.astype(np.float32), proj, cv2.NORM_L2) / proj.shape[0]
        
        if len(all_errors) == 0:  # 只打印第一幅的详细信息
            print(f"\n[调试] {s.base}:")
            print(f"  前3个角点投影: {proj_pts[:3]}")
            print(f"  前3个RGB实测: {rgb_pts[:3]}")
            print(f"  最大误差点索引: {max_err_idx}, 误差: {point_errors[max_err_idx]:.2f}px")
            print(f"  投影范围: X[{proj_pts[:, 0].min():.1f}, {proj_pts[:, 0].max():.1f}], Y[{proj_pts[:, 1].min():.1f}, {proj_pts[:, 1].max():.1f}]")
            print(f"  RGB范围: X[{rgb_pts[:, 0].min():.1f}, {rgb_pts[:, 0].max():.1f}], Y[{rgb_pts[:, 1].min():.1f}, {rgb_pts[:, 1].max():.1f}]")
        
        per_image_errors.append((s.base, float(err)))
        all_errors.append(float(err))

    mean_err = float(np.mean(all_errors)) if all_errors else float("nan")
    return mean_err, per_image_errors, projections


# ----------------------------- 可视化与导出 -----------------------------

def save_overlays_and_errors(
    samples: List[Sample],
    projections: Dict[str, np.ndarray],
    per_image_errors: List[Tuple[str, float]],
    visualize_dir: Optional[Path],
) -> Optional[Path]:
    if visualize_dir is None:
        return None
    visualize_dir.mkdir(parents=True, exist_ok=True)
    # 保存每幅的叠加图
    saved_one: Optional[Path] = None
    for s in samples:
        if s.base not in projections:
            continue
        if s.rgb_image_path is None or not s.rgb_image_path.exists():
            continue
        img = cv2.imread(str(s.rgb_image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        pts_proj = projections[s.base].reshape(-1, 2).astype(np.float32)
        for pt in pts_proj:
            cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 255, 0), -1)
        # 叠加RGB实测角点（红色）用于对比
        if s.rgb_corners is not None:
            for pt in s.rgb_corners.reshape(-1, 2):
                cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 0, 255), -1)
        out_path = visualize_dir / f"overlay_{s.base}.png"
        cv2.imwrite(str(out_path), img)
        if saved_one is None:
            saved_one = out_path

    # 保存误差CSV
    if per_image_errors:
        csv_lines = ["base,mean_reprojection_error_px"]
        for base, e in per_image_errors:
            csv_lines.append(f"{base},{e:.6f}")
        (visualize_dir / "errors.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    return saved_one


def save_rectify_preview(
    sample: Sample,
    K_rgb: np.ndarray,
    dist_rgb: np.ndarray,
    K_tof: np.ndarray,
    dist_tof: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    image_size_rgb: Tuple[int, int],
    image_size_tof: Tuple[int, int],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = 0
    R1, R2, P1, P2, Q, valid1, valid2 = cv2.stereoRectify(
        K_rgb, dist_rgb, K_tof, dist_tof, image_size_rgb, R, T, flags=flags, alpha=0
    )

    map1_rgb = cv2.initUndistortRectifyMap(K_rgb, dist_rgb, R1, P1, image_size_rgb, cv2.CV_16SC2)
    map1_tof = cv2.initUndistortRectifyMap(K_tof, dist_tof, R2, P2, image_size_tof, cv2.CV_16SC2)

    if sample.rgb_image_path and sample.rgb_image_path.exists():
        img_rgb = cv2.imread(str(sample.rgb_image_path), cv2.IMREAD_COLOR)
        rect_rgb = cv2.remap(img_rgb, map1_rgb[0], map1_rgb[1], interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_dir / f"rectify_rgb_{sample.base}.png"), rect_rgb)
    if sample.tof_image_path and sample.tof_image_path.exists():
        img_tof = cv2.imread(str(sample.tof_image_path), cv2.IMREAD_COLOR)
        rect_tof = cv2.remap(img_tof, map1_tof[0], map1_tof[1], interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_dir / f"rectify_tof_{sample.base}.png"), rect_tof)


# ----------------------------- 主流程 -----------------------------

def main():
    parser = argparse.ArgumentParser(description="ToF(IR几何) 与 RGB 外参标定（基于角点文件）")
    parser.add_argument("--rgb-images", type=str, required=True, help="RGB 原图目录")
    parser.add_argument("--tof-images", type=str, required=True, help="ToF 原图目录（用于IR）")
    parser.add_argument("--rgb-corners", type=str, required=True, help="RGB 角点目录（*.npz）")
    parser.add_argument("--tof-corners", type=str, required=True, help="ToF 角点目录（*.npz）")
    parser.add_argument("--inner-corners-x", type=int, required=True)
    parser.add_argument("--inner-corners-y", type=int, required=True)
    parser.add_argument("--square-size-mm", type=float, required=True)
    parser.add_argument("--rgb-calib-json", type=str, default=None)
    parser.add_argument("--tof-calib-json", type=str, default=None)
    parser.add_argument("--fix-intrinsic", action="store_true", help="若两侧提供JSON，则固定内参仅解外参")
    parser.add_argument("--visualize-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="stereo_extrinsics.json")
    parser.add_argument("--rectify-preview", action="store_true")

    args = parser.parse_args()

    rgb_img_dir = Path(args.__dict__["--rgb-images"] if "--rgb-images" in args.__dict__ else args.rgb_images)
    tof_img_dir = Path(args.__dict__["--tof-images"] if "--tof-images" in args.__dict__ else args.tof_images)
    rgb_corners_dir = Path(args.__dict__["--rgb-corners"] if "--rgb-corners" in args.__dict__ else args.rgb_corners)
    tof_corners_dir = Path(args.__dict__["--tof-corners"] if "--tof-corners" in args.__dict__ else args.tof_corners)
    visualize_dir = Path(args.visualize_dir) if args.visualize_dir else None
    output_path = Path(args.output)

    inner_x = args.inner_corners_x
    inner_y = args.inner_corners_y
    square_mm = args.square_size_mm

    # 1) 扫描并配对样本
    rgb_map = scan_bases(rgb_corners_dir)
    tof_map = scan_bases(tof_corners_dir)
    rgb_bases = set(rgb_map.keys())
    tof_bases = set(tof_map.keys())
    inter = sorted(list(rgb_bases & tof_bases))
    missing_rgb = sorted(list(tof_bases - rgb_bases))
    missing_tof = sorted(list(rgb_bases - tof_bases))

    print(f"RGB角点: {len(rgb_bases)} 个, ToF角点: {len(tof_bases)} 个, 配对: {len(inter)} 个")
    if missing_rgb:
        print("[WARN] 缺少 RGB 角点样本: ", ", ".join(missing_rgb[:20]) + (" ..." if len(missing_rgb) > 20 else ""))
    if missing_tof:
        print("[WARN] 缺少 ToF 角点样本: ", ", ".join(missing_tof[:20]) + (" ..." if len(missing_tof) > 20 else ""))

    samples: List[Sample] = []
    skipped = 0
    for base in inter:
        c_rgb = load_corners_npz(rgb_map[base])
        c_tof = load_corners_npz(tof_map[base])
        if c_rgb is None or c_tof is None:
            skipped += 1
            continue
        if c_rgb.shape[0] != c_tof.shape[0]:
            print(f"[WARN] 角点数量不匹配：{base}: rgb={c_rgb.shape[0]} tof={c_tof.shape[0]}")
            skipped += 1
            continue
        rgb_img = find_image_by_base(rgb_img_dir, base)
        tof_img = find_image_by_base(tof_img_dir, base)
        samples.append(Sample(base=base, rgb_corners=c_rgb, tof_corners=c_tof, rgb_image_path=rgb_img, tof_image_path=tof_img))

    if len(samples) < 3:
        raise ValueError(f"有效配对样本不足（≥3）。有效: {len(samples)}, 被跳过: {skipped}")

    # 2) 读取图像尺寸
    # 优先尝试第一对样本的同名图像，否则各自目录第一张
    first_base = samples[0].base
    rgb_size = ensure_image_size(rgb_img_dir, first_base)
    tof_size = ensure_image_size(tof_img_dir, first_base)
    print(f"RGB 图像尺寸: {rgb_size}, ToF 图像尺寸: {tof_size}")

    # 3) 读取或估计两路内参
    rgb_intr = load_intrinsics_json(Path(args.rgb_calib_json)) if args.rgb_calib_json else None
    tof_intr = load_intrinsics_json(Path(args.tof_calib_json)) if args.tof_calib_json else None

    if rgb_intr is None:
        print("[INFO] 未提供 RGB 内参，基于角点估计...")
        # 仅用有有效RGB角点的样本
        rgb_K, rgb_dist = estimate_intrinsics_from_corners(samples, inner_x, inner_y, square_mm, rgb_size)
    else:
        rgb_K, rgb_dist = rgb_intr

    if tof_intr is None:
        print("[INFO] 未提供 ToF 内参，基于角点估计...")
        tof_K, tof_dist = estimate_intrinsics_from_corners_tof(samples, inner_x, inner_y, square_mm, tof_size)
    else:
        tof_K, tof_dist = tof_intr

    fix_intrinsic = bool(args.fix_intrinsic and rgb_intr is not None and tof_intr is not None)
    if fix_intrinsic:
        print("[INFO] 使用 CALIB_FIX_INTRINSIC 仅解外参")
    else:
        print("[INFO] 使用 CALIB_USE_INTRINSIC_GUESS，允许微调并固定高阶畸变")

    # 4) 立体标定
    (K_rgb_out, dist_rgb_out, R, T), (K_tof_out, dist_tof_out, E, F) = stereo_calibrate(
        samples,
        inner_x,
        inner_y,
        square_mm,
        rgb_K,
        rgb_dist,
        tof_K,
        tof_dist,
        rgb_size,
        fix_intrinsic,
    )

    print("立体标定完成。")
    print("R (ToF→RGB):\n", R)
    print("t (ToF→RGB):\n", T.reshape(-1))

    # 5) 计算立体标定本身的逐图重投影误差（更可靠）
    objp = build_object_points(inner_x, inner_y, square_mm).astype(np.float32)
    stereo_errors_rgb = []
    stereo_errors_tof = []
    for s in samples:
        if s.rgb_corners.shape[0] != inner_x * inner_y or s.tof_corners.shape[0] != inner_x * inner_y:
            continue
        # 用stereoCalibrate找到的R和T计算ToF→RGB投影
        # 先在ToF上求解棋盘位姿
        ok, rvec_tof, tvec_tof = cv2.solvePnP(
            objp, s.tof_corners, K_tof_out, dist_tof_out, flags=cv2.SOLVEPNP_IPPE if hasattr(cv2, "SOLVEPNP_IPPE") else 0
        )
        if not ok:
            continue
        # 将棋盘3D点从ToF系转换到RGB系
        R_tof, _ = cv2.Rodrigues(rvec_tof)
        Xb_T = objp.T
        tvec_tof = tvec_tof.reshape(3, 1)
        T_vec = T.reshape(3, 1)
        X_tof = R_tof @ Xb_T + tvec_tof
        X_rgb = R @ X_tof + T_vec
        X_rgb = X_rgb.T.reshape(-1, 1, 3)
        # 投影到RGB
        proj_rgb, _ = cv2.projectPoints(X_rgb, np.zeros((3, 1)), np.zeros((3, 1)), K_rgb_out, dist_rgb_out)
        err_rgb = cv2.norm(s.rgb_corners.astype(np.float32), proj_rgb.astype(np.float32), cv2.NORM_L2) / len(proj_rgb)
        stereo_errors_rgb.append(float(err_rgb))
        # ToF本身的重投影（验证ToF内参）
        proj_tof, _ = cv2.projectPoints(objp.reshape(-1, 1, 3), rvec_tof, tvec_tof, K_tof_out, dist_tof_out)
        err_tof = cv2.norm(s.tof_corners.astype(np.float32), proj_tof.astype(np.float32), cv2.NORM_L2) / len(proj_tof)
        stereo_errors_tof.append(float(err_tof))
    
    if stereo_errors_rgb:
        print(f"\n立体标定逐图重投影误差:")
        print(f"  RGB (ToF→RGB投影): 均值 {np.mean(stereo_errors_rgb):.2f} px, 范围 [{np.min(stereo_errors_rgb):.2f}, {np.max(stereo_errors_rgb):.2f}] px")
        print(f"  ToF (本身): 均值 {np.mean(stereo_errors_tof):.2f} px, 范围 [{np.min(stereo_errors_tof):.2f}, {np.max(stereo_errors_tof):.2f}] px")

    # 6) 误差评估（PnP on ToF, project to RGB）
    mean_err, per_img_errors, projections = mean_reprojection_error_via_pnp(
        samples,
        inner_x,
        inner_y,
        square_mm,
        K_rgb_out,
        dist_rgb_out,
        K_tof_out,
        dist_tof_out,
        R,
        T,
    )

    if not np.isnan(mean_err):
        print(f"Mean reprojection error (RGB px): {mean_err:.4f}")
    else:
        print("[WARN] 无法计算重投影误差")

    # 6) 可视化
    saved_overlay = save_overlays_and_errors(samples, projections, per_img_errors, visualize_dir)
    if saved_overlay:
        print(f"已保存叠加预览: {saved_overlay}")

    # 7) 输出 JSON
    result = {
        "K_rgb": K_rgb_out.tolist(),
        "dist_rgb": dist_rgb_out.reshape(-1, 1).tolist(),
        "K_tof": K_tof_out.tolist(),
        "dist_tof": dist_tof_out.reshape(-1, 1).tolist(),
        "R": R.tolist(),
        "t": T.reshape(-1, 1).tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "image_width_rgb": int(rgb_size[0]),
        "image_height_rgb": int(rgb_size[1]),
        "image_width_tof": int(tof_size[0]),
        "image_height_tof": int(tof_size[1]),
        "inner_corners": [int(inner_x), int(inner_y)],
        "square_size_mm": float(square_mm),
        "mean_reprojection_error_px": float(mean_err) if not np.isnan(mean_err) else None,
        "per_image_errors": [{"base": b, "mean_px": float(e)} for b, e in per_img_errors],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {output_path}")

    # 8) 可选 rectify 预览
    if args.rectify_preview:
        if visualize_dir is None:
            print("[INFO] 未指定 --visualize-dir，使用输出文件目录保存rectify预览。")
            visualize_base = output_path.parent
        else:
            visualize_base = visualize_dir
        sample0 = samples[0]
        save_rectify_preview(sample0, K_rgb_out, dist_rgb_out, K_tof_out, dist_tof_out, R, T, rgb_size, tof_size, visualize_base)
        print("已保存 rectify 预览图。")


if __name__ == "__main__":
    main()


