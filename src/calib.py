"""Phenikaa calibration loader. Extrinsics map LiDAR -> camera; ego frame = LiDAR frame."""

import json
from pathlib import Path

import numpy as np

from .contracts import CamCalib

PHENIKAA_CAMERAS = ("CAM_P_F", "CAM_P_FL", "CAM_P_FR", "CAM_P_B", "CAM_P_L", "CAM_P_R")


def load_phenikaa_calib(calib_dir, z_ground=-0.04):
    calib_dir = Path(calib_dir)
    intr = json.loads((calib_dir / "Camera_Intrinsics.json").read_text())
    extr = json.loads((calib_dir / "Sensor_Extrinsics.json").read_text())
    out = {}
    for name in PHENIKAA_CAMERAS:
        c = intr[name]
        ext = np.asarray(extr[name], dtype=float).reshape(4, 4)
        out[name] = CamCalib(c["camera_matrix"], c["distortion_coefficients"],
                             (c["image_width"], c["image_height"]), np.linalg.inv(ext), z_ground)
    return out
