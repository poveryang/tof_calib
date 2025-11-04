# ToF-RGB 外参标定项目

## 📋 项目概述

本项目用于标定 ToF 相机（使用 IR 图像）与 RGB 相机的外参，实现两个相机之间的 3D 坐标变换。

**最终标定精度**：
- 立体标定 RMS：0.61 px
- ToF→RGB 投影误差：1.55 px（误差范围 0.88-2.47 px）

---

## 🚀 快速开始

### 前提条件
- 棋盘：6x5 方格（5x4 内角点），单格 40mm
- 采集 15 张以上配对的 RGB 和 ToF 图像
- **重要**：ToF 和 RGB 相机视角相差约 90 度

### 完整流程（7 步）

#### 1. 检测 RGB 角点
```bash
python auto_detect_rgb_corners.py \
  --images-dir data/rgb_images \
  --corners-dir data/rgb_corners \
  --inner-corners-x 5 \
  --inner-corners-y 4 \
  --save-vis data/rgb_corners_vis
```

#### 2. 旋转 ToF 图像（关键！）
```bash
python << 'EOF'
import cv2
from pathlib import Path

tof_dir = Path('data/tof_images')
tof_rotated_dir = Path('data/tof_images_rotated')
tof_rotated_dir.mkdir(parents=True, exist_ok=True)

for img_path in sorted(tof_dir.glob('*.png')):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(str(tof_rotated_dir / img_path.name), img_rotated)
    print(f"✓ {img_path.name}")
EOF
```

#### 3. 标注 ToF 角点（手动，在旋转后的图像上）
```bash
python manual_corner_labeling.py label \
  --images-dir data/tof_images_rotated \
  --corners-dir data/tof_corners_rotated \
  --inner-corners-x 5 \
  --inner-corners-y 4 \
  --scale 12.0 \
  --save-vis data/tof_corners_vis_rotated
```
**标注要点**：从左上到右下，行优先顺序

#### 4. 诊断角点对应关系
```bash
python diagnose_corner_correspondence.py \
  --rgb-image data/rgb_images/0.png \
  --tof-image data/tof_images_rotated/0.png \
  --rgb-corners data/rgb_corners/0_corners.npz \
  --tof-corners data/tof_corners_rotated/0_corners.npz \
  --output-dir data/diagnosis

# 查看结果，确认 0 号和 19 号点都匹配
open data/diagnosis/rgb_order.png data/diagnosis/tof_order.png
```

#### 5. 标定 RGB 内参
```bash
python calibrate_mono.py \
  --images-dir data/rgb_images \
  --corners-dir data/rgb_corners \
  --inner-corners-x 5 \
  --inner-corners-y 4 \
  --square-size-mm 40.0 \
  --output data/rgb_calib.json
```

#### 6. 标定 ToF 内参
```bash
python manual_corner_labeling.py calibrate \
  --images-dir data/tof_images_rotated \
  --corners-dir data/tof_corners_rotated \
  --inner-corners-x 5 \
  --inner-corners-y 4 \
  --square-size-mm 40.0 \
  --output data/tof_calib_rotated.json
```

#### 7. 标定外参
```bash
python calibrate_extrinsics_from_corners.py \
  --rgb-images data/rgb_images \
  --tof-images data/tof_images_rotated \
  --rgb-corners data/rgb_corners \
  --tof-corners data/tof_corners_rotated \
  --inner-corners-x 5 \
  --inner-corners-y 4 \
  --square-size-mm 40.0 \
  --rgb-calib-json data/rgb_calib.json \
  --tof-calib-json data/tof_calib_rotated.json \
  --fix-intrinsic \
  --visualize-dir data/vis_corrected \
  --output data/stereo_extrinsics.json
```

---

## 📁 项目结构

```
tof_calib/
├── 核心脚本
│   ├── auto_detect_rgb_corners.py           # RGB角点自动检测
│   ├── manual_corner_labeling.py            # ToF角点手动标注+标定
│   ├── calibrate_mono.py                    # 单目内参标定
│   ├── calibrate_extrinsics_from_corners.py # 立体外参标定
│   ├── diagnose_corner_correspondence.py    # 角点对应诊断
│   ├── detect_chessboard_corners.py         # 角点检测底层库
│   └── generate_chessboard.py               # 棋盘生成工具
│
└── data/
    ├── rgb_images/              # RGB原始图像
    ├── rgb_corners/             # RGB角点
    ├── rgb_calib.json           # RGB内参 ✅
    ├── tof_images/              # ToF原始图像
    ├── tof_images_rotated/      # ToF旋转后图像 ⚠️ 用于标注
    ├── tof_corners_rotated/     # ToF角点
    ├── tof_calib_rotated.json   # ToF内参 ✅
    ├── stereo_extrinsics.json   # 外参结果 ✅
    └── vis_corrected/           # 叠加可视化
```

---

## ✅ 验证结果

### 查看叠加图
```bash
open data/vis_corrected/overlay_0.png
```
绿色点（ToF 投影）应该与红色点（RGB 实测）完全重合

### 查看误差统计
```bash
cat data/vis_corrected/errors.csv
```
每张图误差应该在 0.5-3 px

### 查看外参
```python
import json
import numpy as np

with open('data/stereo_extrinsics.json') as f:
    extr = json.load(f)

R = np.array(extr['R'])
T = np.array(extr['t']).flatten()

print(f"R (ToF→RGB):\n{R}")
print(f"t (ToF→RGB): {T} mm")
print(f"平均投影误差: {extr['mean_reprojection_error_px']:.2f} px")
```

---

## 🎯 关键要点

### 1. ToF 图像必须旋转
- ToF 和 RGB 视角相差 90 度
- **必须先将 ToF 图像逆时针旋转 90 度**
- 然后在旋转后的图像上标注角点

### 2. 角点顺序必须一致
- RGB 和 ToF 的 0 号点必须在同一物理位置
- 扫描方向必须一致（都是行优先：从左到右，从上到下）
- 使用 `diagnose_corner_correspondence.py` 验证

### 3. stereoCalibrate 参数顺序
```python
# 正确的顺序（已修正）
cv2.stereoCalibrate(
    objpoints,
    imgpoints_tof,    # ToF 在前
    imgpoints_rgb,    # RGB 在后
    K_tof, dist_tof,
    K_rgb, dist_rgb,
    ...
)
# 返回的 R, T 是 ToF→RGB
```

### 4. ToF 角点标注质量
- 放大 12 倍后仔细标注
- 点击角点精确中心
- 确保相邻点间距均匀
- 标注质量直接影响标定精度

---

## 🐛 常见问题

### Q: RGB 角点检测失败？
**A**: 添加预处理参数
```bash
--eq-hist --blur 7
```

### Q: 外参投影误差很大？
**A**: 检查清单：
1. ToF 图像是否正确旋转（逆时针 90 度）？
2. ToF 角点是否在旋转后的图像上标注？
3. 角点对应关系是否正确（用诊断工具检查）？
4. ToF 角点标注质量是否足够好？

### Q: 如何判断标定成功？
**A**: 
- 立体标定 RMS < 1 px
- ToF→RGB 投影误差 < 3 px
- 叠加图中绿点和红点重合

---

## 📖 使用标定结果

### 坐标变换
```python
import json
import numpy as np

# 加载外参
with open('data/stereo_extrinsics.json') as f:
    extr = json.load(f)

R = np.array(extr['R'])
T = np.array(extr['t']).reshape(3, 1)

# 将 ToF 相机坐标系中的 3D 点变换到 RGB 相机坐标系
X_tof = np.array([[x, y, z]]).T  # (3, N)
X_rgb = R @ X_tof + T

# 投影到 RGB 图像
K_rgb = np.array(extr['K_rgb'])
dist_rgb = np.array(extr['dist_rgb']).flatten()

import cv2
proj_rgb, _ = cv2.projectPoints(
    X_rgb.T.reshape(-1, 1, 3),
    np.zeros((3, 1)),
    np.zeros((3, 1)),
    K_rgb,
    dist_rgb
)
```

### 注意事项
1. **ToF 坐标系**：基于旋转后的图像（101 x 100）
2. **RGB 坐标系**：原始图像（2600 x 1952）
3. **平移单位**：毫米（mm）

---

## 📞 技术支持

遇到问题可参考：
1. 检查 `data/vis_corrected/` 中的叠加图
2. 查看 `errors.csv` 确认误差分布
3. 使用诊断工具验证角点对应
4. 检查内参和外参的数值是否合理

---

## 🎓 核心脚本说明

### auto_detect_rgb_corners.py
- RGB 角点自动检测
- 自动调整顺序（0 号点在左上）
- 支持预处理和可视化

### manual_corner_labeling.py
- **label 模式**：手动标注 ToF 角点
- **calibrate 模式**：从角点文件标定内参
- 支持图像放大和可视化保存

### calibrate_mono.py
- 单目相机内参标定
- 优先使用已有角点文件
- 避免重复检测

### calibrate_extrinsics_from_corners.py
- 立体外参标定（核心）
- 计算 R, T, E, F
- 误差评估和可视化
- **已修正 stereoCalibrate 参数顺序**

### diagnose_corner_correspondence.py
- 可视化角点顺序和编号
- 验证两路角点对应关系
- 帮助发现顺序问题

### detect_chessboard_corners.py
- 底层角点检测库
- 多种检测算法
- 被其他脚本调用

### generate_chessboard.py
- 生成打印用的标定棋盘

---

## 📊 标定结果文件

### stereo_extrinsics.json
```json
{
  "K_rgb": [...],                      // RGB内参矩阵
  "dist_rgb": [...],                   // RGB畸变系数
  "K_tof": [...],                      // ToF内参矩阵（旋转后）
  "dist_tof": [...],                   // ToF畸变系数
  "R": [...],                          // 旋转矩阵 (3x3, ToF→RGB)
  "t": [...],                          // 平移向量 (3x1, mm, ToF→RGB)
  "E": [...],                          // 本质矩阵
  "F": [...],                          // 基础矩阵
  "mean_reprojection_error_px": 1.55,  // 平均投影误差
  "per_image_errors": [...]            // 逐图误差统计
}
```

---

## ⚠️ 重要注意事项

### 1. ToF 图像旋转
由于 ToF 和 RGB 相机视角相差 90 度，**必须先将 ToF 图像逆时针旋转 90 度**，然后：
- 在旋转后的图像上标注角点
- 用旋转后的图像和角点标定 ToF 内参
- 外参标定时使用旋转后的 ToF 图像和内参

### 2. 角点对应关系
- 0 号点必须在同一物理位置（左上角）
- 19 号点必须在同一物理位置（右下角）
- 扫描顺序必须一致（都是行优先）
- 使用 `diagnose_corner_correspondence.py` 验证

### 3. 标注质量要求
- 放大 12 倍后精确点击角点中心
- 相邻点间距应该均匀（标准差 < 2 px）
- 标注质量直接影响内参和外参精度

---

## 📁 数据目录结构

```
data/
├── rgb_images/              # RGB原始图像 (2600x1952)
├── rgb_corners/             # RGB角点文件 (.npz)
├── rgb_corners_vis/         # RGB角点可视化
├── rgb_calib.json           # RGB内参 ✅
│
├── tof_images/              # ToF原始图像 (100x101)
├── tof_images_rotated/      # ToF旋转后 (101x100) ⚠️ 用于标注
├── tof_corners_rotated/     # ToF角点文件
├── tof_corners_vis_rotated/ # ToF角点可视化
├── tof_calib_rotated.json   # ToF内参 ✅
│
├── stereo_extrinsics.json   # 外参结果 ✅
└── vis_corrected/           # 外参验证
    ├── overlay_*.png        # 叠加图（绿=ToF投影，红=RGB实测）
    └── errors.csv           # 误差统计
```

---

## 🔧 故障排除

| 问题 | 解决方案 |
|------|----------|
| RGB 角点检测失败 | 添加 `--eq-hist --blur 7` |
| ToF 图像太小看不清 | 使用 `--scale 12.0` 放大标注 |
| 投影误差很大（>10px） | 检查角点对应关系和 ToF 图像是否旋转 |
| 内参焦距异常 | 重新标注 ToF 角点，确保间距均匀 |
| 0号点和19号点不匹配 | 用 `diagnose_corner_correspondence.py` 诊断 |

---

## 🎯 成功标定的指标

- ✅ RGB 单目重投影误差 < 0.5 px
- ✅ ToF 单目重投影误差 < 0.5 px
- ✅ 立体标定 RMS < 1 px
- ✅ ToF→RGB 投影误差 < 3 px
- ✅ 叠加图中绿点和红点重合

---

## 💡 关键教训

1. **ToF 图像必须旋转**：两相机视角相差 90 度
2. **角点顺序必须一致**：0 号点和 19 号点对应同一物理位置
3. **stereoCalibrate 参数顺序很重要**：ToF 在前，RGB 在后
4. **标注质量很关键**：手动标注要精确，间距要均匀
5. **每步都要验证**：使用诊断工具和可视化验证结果

---

## 📚 扩展使用

### 生成标定棋盘
```bash
python generate_chessboard.py \
  --cols 6 \
  --rows 5 \
  --square-size 40 \
  --output chessboard.pdf
```

### 从已有角点重新标定
```bash
# RGB 内参
python calibrate_mono.py \
  --images-dir data/rgb_images \
  --corners-dir data/rgb_corners \
  --inner-corners-x 5 --inner-corners-y 4 \
  --square-size-mm 40.0 \
  --output data/rgb_calib_new.json

# ToF 内参
python manual_corner_labeling.py calibrate \
  --images-dir data/tof_images_rotated \
  --corners-dir data/tof_corners_rotated \
  --inner-corners-x 5 --inner-corners-y 4 \
  --square-size-mm 40.0 \
  --output data/tof_calib_new.json
```

---

## 🎉 项目完成

- ✅ RGB 角点自动检测
- ✅ ToF 角点手动标注工具
- ✅ 单目内参标定
- ✅ 立体外参标定
- ✅ 误差评估和可视化
- ✅ 完整文档

**最终精度：1.55 px** 🎯

