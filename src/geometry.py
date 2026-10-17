"""Pixel <-> ego-frame geometry (x forward, y left, z up, meters)."""

import cv2
import numpy as np

from .contracts import CamCalib, Hit


def _ray(u, v, cam: CamCalib):
    """Undistorted camera-frame ray (z=1) for a pixel."""
    pt = np.array([[[u, v]]], dtype=np.float64)
    xn, yn = cv2.undistortPoints(pt, cam.K, cam.dist)[0, 0]
    return np.array([xn, yn, 1.0])


def ground_point(xyxy, cam: CamCalib) -> Hit | None:
    x1, y1, x2, y2 = map(float, xyxy)
    u, v = (x1 + x2) / 2, y2
    ray_c = _ray(u, v, cam)
    d = cam.T_cam2ego[:3, :3] @ ray_c
    o = cam.T_cam2ego[:3, 3]
    if d[2] >= -1e-9:
        return None
    s = (cam.z_ground - o[2]) / d[2]
    if s <= 0:
        return None
    p = o + s * d
    at_edge = v >= cam.size[1] - 2 or v <= 1
    return Hit(float(p[0]), float(p[1]), float(np.hypot(p[0], p[1])), "ground", bool(at_edge))


def height_point(xyxy, cam: CamCalib, H: float) -> Hit | None:
    x1, y1, x2, y2 = map(float, xyxy)
    h = y2 - y1
    if h <= 1e-6 or H <= 0:
        return None
    ray_c = _ray((x1 + x2) / 2, (y1 + y2) / 2, cam)
    depth = cam.K[1, 1] * H / h  # metric distance along the undistorted ray
    p = cam.T_cam2ego[:3, :3] @ (ray_c / np.linalg.norm(ray_c) * depth) + cam.T_cam2ego[:3, 3]
    return Hit(float(p[0]), float(p[1]), float(np.hypot(p[0], p[1])), "height", False)


def project_ego(points_ego, cam: CamCalib):
    """Nx3 ego points -> (Nx2 pixels, bool valid mask: in front of camera and inside image)."""
    P = np.asarray(points_ego, dtype=float).reshape(-1, 3)
    T = np.linalg.inv(cam.T_cam2ego)
    pc = P @ T[:3, :3].T + T[:3, 3]
    front = pc[:, 2] > 0.1
    safe = pc.copy()
    safe[~front, 2] = 1.0
    px, _ = cv2.projectPoints(safe.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.dist)
    px = px.reshape(-1, 2)
    w, h = cam.size
    valid = front & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    return px, valid
