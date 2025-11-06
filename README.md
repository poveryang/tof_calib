# ToF ↔ RGB Calibration Toolkit

This repository provides a compact toolchain for calibrating a depth (ToF) camera with an RGB camera using chessboard captures.  The refactor focuses on a small set of reusable modules with thin command-line wrappers.

## 📦 Project layout

```
.
├── tof_calib/                 # Core Python package
│   ├── chessboard.py          # Chessboard patterns and corner handling
│   ├── calibration.py         # Mono/stereo calibration helpers
│   ├── labeling.py            # Minimal interactive labelling helper
│   └── projection.py          # Depth-to-RGB projection utilities
├── scripts/                   # Command line entry points
│   ├── detect_corners.py      # Batch corner detection
│   ├── label_corners.py       # Manual labelling for difficult frames
│   ├── calibrate_intrinsics.py# Monocular calibration
│   ├── calibrate_stereo.py    # Stereo (extrinsic) calibration
│   └── project_depth.py       # Project depth samples into RGB
└── data/                      # Example data layout (not required)
```

## 🔁 Typical workflow

1. **Prepare the chessboard configuration** – Specify the number of inner corners (`columns × rows`) and the physical square size in millimetres.  These options are shared across all scripts.

2. **Label or detect corners**
   - Automatic detection for RGB or clean ToF frames:
     ```bash
     python scripts/detect_corners.py data/rgb_images data/rgb_corners \
       --columns 5 --rows 4 --square-size 40
     ```
   - Manual labelling for ToF frames that require supervision:
     ```bash
     python scripts/label_corners.py data/tof_images_rotated data/tof_corners \
       --columns 5 --rows 4 --square-size 40 --scale 12
     ```

3. **Calibrate intrinsics for each camera**
   ```bash
   python scripts/calibrate_intrinsics.py data/rgb_images data/rgb_calib.json \
     --columns 5 --rows 4 --square-size 40 --corners data/rgb_corners

   python scripts/calibrate_intrinsics.py data/tof_images_rotated data/tof_calib.json \
     --columns 5 --rows 4 --square-size 40 --corners data/tof_corners
   ```

4. **Solve the stereo extrinsics**
   ```bash
   python scripts/calibrate_stereo.py \
     data/tof_images_rotated data/rgb_images \
     data/tof_calib.json data/rgb_calib.json \
     data/stereo_extrinsics.json \
     --columns 5 --rows 4 --square-size 40 \
     --corners-a data/tof_corners --corners-b data/rgb_corners
   ```

5. **Project a depth capture into the RGB camera**
   ```bash
   python scripts/project_depth.py data/test/depth.csv data/test/rgb.png \
     data/stereo_extrinsics.json --depth-camera a --depth-scale 2.4 \
     --rotate-k 1 --pad-last-row 3400 --output data/test/projection.png
   ```

Each script prints a concise summary and writes artefacts (corner files, calibration JSONs, overlay images) to the specified output paths.

## 🧠 Key concepts

- **ChessboardPattern** encapsulates the geometric description of the calibration board and can generate object points for OpenCV calibration routines.
- **CornerDetectionConfig** centralises image preprocessing options (CLAHE, blurring, upscaling, etc.) used by automatic detectors.
- **IntrinsicCalibrationResult** and **StereoCalibrationResult** wrap OpenCV outputs and provide helpers for saving/loading JSON representations.
- **project_depth_to_rgb** converts a dense depth map into 3D points, applies the extrinsic transform, and reprojects the surviving points back into the RGB camera.

The refactor isolates reusable logic inside `tof_calib/` so additional applications (e.g. notebooks or alternative front-ends) can import the same core functionality without relying on command-line scripts.
