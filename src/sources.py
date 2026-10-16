"""Backend-owned nuScenes and webcam readers. PhenikaaSource belongs to Core."""

import json
import time
import warnings
from pathlib import Path

import numpy as np

from .contracts import CamCalib, Frame

NUSCENES_CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


def quaternion_matrix(q):
    """Convert a nuScenes quaternion in w, x, y, z order into a rotation."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Rotation quaternion must be nonzero")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def pose_matrix(record):
    transform = np.eye(4)
    transform[:3, :3] = quaternion_matrix(record["rotation"])
    transform[:3, 3] = record["translation"]
    return transform


def _gt_class(category):
    if category.startswith("human.pedestrian"):
        return "person"
    if category in ("vehicle.motorcycle", "vehicle.bicycle"):
        return "rider"
    if category == "vehicle.car" or category.startswith("vehicle.bus") or category == "vehicle.truck":
        return "car"
    return None


class NuScenesSource:
    """Read synchronized keyframes in the reference LIDAR_TOP ego frame.

    Camera extrinsics include the ego motion between each camera exposure and
    the reference exposure. Ground truth and all camera rays share that frame.
    """

    def __init__(self, root, scene=None, version="v1.0-mini", cameras=None, nusc=None):
        if nusc is None:
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
        self.nusc = nusc
        self.root = Path(root)
        self.cameras = tuple(cameras or NUSCENES_CAMERAS)
        scenes = nusc.scene
        selected = next((s for s in scenes if scene in (s["name"], s["token"])), None) if scene else scenes[0]
        if selected is None:
            raise ValueError(f"Unknown scene {scene!r}; available: {[s['name'] for s in scenes]}")
        self.scene = selected["name"]
        self.samples = []
        token = selected["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            self.samples.append(sample)
            token = sample["next"]
        self.index = 0

    def __len__(self):
        return len(self.samples)

    def seek(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        self.index = index

    def __iter__(self):
        return self

    def __next__(self):
        import cv2
        if self.index >= len(self):
            raise StopIteration
        index = self.index
        sample = self.samples[index]
        reference = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_T = pose_matrix(self.nusc.get("ego_pose", reference["ego_pose_token"]))
        world_to_ego = np.linalg.inv(ego_T)
        images, calibration = {}, {}
        for name in self.cameras:
            data = self.nusc.get("sample_data", sample["data"][name])
            path = self.root / data["filename"]
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Cannot read camera image: {path}")
            sensor = self.nusc.get("calibrated_sensor", data["calibrated_sensor_token"])
            camera_ego = pose_matrix(self.nusc.get("ego_pose", data["ego_pose_token"]))
            transform = world_to_ego @ camera_ego @ pose_matrix(sensor)
            images[name] = image
            calibration[name] = CamCalib(sensor["camera_intrinsic"], np.zeros(5),
                                        (image.shape[1], image.shape[0]), transform, 0.0)
        gt = []
        for token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", token)
            cls = _gt_class(ann["category_name"])
            if cls is None:
                continue
            center = world_to_ego @ np.r_[ann["translation"], 1.0]
            rotation = world_to_ego[:3, :3] @ quaternion_matrix(ann["rotation"])
            width, length, height = ann["size"]
            gt.append(dict(cls=cls, x=float(center[0]), y=float(center[1]), z=float(center[2]),
                           l=float(length), w=float(width), h=float(height),
                           yaw=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                           visibility=int(ann.get("visibility_token", 0)), token=token,
                           instance_token=ann.get("instance_token")))
        self.index += 1
        return Frame(sample["timestamp"] / 1e6, images, calibration, ego_T, gt, index)

    def close(self):
        pass


def load_webcam_calib(path, size):
    """Read the shared camera contract or explicitly flag the HFOV fallback."""
    width, height = size
    camera_to_ego = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    if path and Path(path).is_file():
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream)
        K = np.asarray(data["K"], dtype=float).reshape(3, 3)
        old_width, old_height = data.get("size", size)
        K[0, :] *= width / old_width
        K[1, :] *= height / old_height
        return CamCalib(K, data.get("dist", [0]*5), size,
                        data.get("T_cam2ego", camera_to_ego), data.get("z_ground", 0)), bool(data.get("estimated", False))
    focal = (width / 2) / np.tan(np.deg2rad(70 / 2))
    warnings.warn("Webcam calibration missing: using estimated HFOV=70 degrees", stacklevel=2)
    K = [[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]]
    return CamCalib(K, np.zeros(5), size, camera_to_ego), True


class WebcamSource:
    scene = "live"

    def __init__(self, device=0, calib_path="calib/webcam.json", width=None, height=None):
        import cv2
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open webcam {device}")
        if width:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.calib_path = calib_path
        self.calibration = None
        self.estimated_calib = False
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        ok, image = self.capture.read()
        captured = time.perf_counter()
        if not ok:
            raise RuntimeError("Webcam frame capture failed")
        size = (image.shape[1], image.shape[0])
        if self.calibration is None or self.calibration.size != size:
            self.calibration, self.estimated_calib = load_webcam_calib(self.calib_path, size)
        result = Frame(captured, {"CAM_FRONT": image}, {"CAM_FRONT": self.calibration},
                       np.eye(4), None, self.index)
        self.index += 1
        return result

    def close(self):
        self.capture.release()


_PHENIKAA_CLASS = {"Rider": "rider", "Pedestrian": "person", "Car": "car", "Bus": "car", "Truck": "car"}


def _stamp(path):
    sec, nsec = Path(path).stem.split("-")
    return int(sec) + int(nsec) / 1e9


class PhenikaaSource:
    """Six synchronized Phenikaa cameras; CAM_P_F timestamps are the master clock.

    Label z is the box centre. Ego frame = LiDAR frame.
    """

    frame_period_s = 0.1
    MASTER = "CAM_P_F"
    TOLERANCE_S = 0.05

    def __init__(self, root, scene="COMPLEX CITY", z_ground=-0.04, cameras=None):
        from .calib import PHENIKAA_CAMERAS, load_phenikaa_calib
        self.root = Path(root)
        self.scene = scene
        self.dir = self.root / scene
        self.cameras = tuple(cameras or PHENIKAA_CAMERAS)
        self.calib = load_phenikaa_calib(self.root / "Calib Information", z_ground)
        files = {}
        for cam in self.cameras:
            files[cam] = sorted((_stamp(p), p) for p in (self.dir / "camera" / cam).glob("*.jpg")
                                if "(" not in p.name)
        self.timestamps = [t for t, _ in files[self.MASTER]]
        self.rows = []  # per frame: {cam: path}
        stamps = {cam: np.array([x[0] for x in items]) for cam, items in files.items()}
        for k, t in enumerate(self.timestamps):
            row = {self.MASTER: files[self.MASTER][k][1]}
            for cam in self.cameras:
                if cam == self.MASTER:
                    continue
                j = int(np.argmin(np.abs(stamps[cam] - t)))
                if abs(stamps[cam][j] - t) > self.TOLERANCE_S:
                    raise ValueError(f"{cam}: no frame within 50 ms of {t:.3f}")
                row[cam] = files[cam][j][1]
            self.rows.append(row)
        self._pose = self._stamped("ego_pose")
        self._label = self._stamped("label")
        self.index = 0

    def _stamped(self, sub):
        return sorted((_stamp(p), p) for p in (self.dir / sub).glob("*.txt") if "(" not in p.name)

    @staticmethod
    def _nearest(items, t):
        if not items:
            return None
        j = int(np.argmin([abs(x[0] - t) for x in items]))
        return items[j][1] if abs(items[j][0] - t) <= 0.05 else None

    def __len__(self):
        return len(self.rows)

    def seek(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        self.index = index

    def __iter__(self):
        return self

    def __next__(self):
        import cv2
        if self.index >= len(self):
            raise StopIteration
        i = self.index
        t = self.timestamps[i]
        images = {}
        for cam, path in self.rows[i].items():
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"Cannot read camera image: {path}")
            images[cam] = img
        ego_T = np.eye(4)
        pp = self._nearest(self._pose, t)
        if pp is not None:
            ego_T = np.vstack([np.loadtxt(pp).reshape(3, 4), [0, 0, 0, 1]])
        gt = []
        lp = self._nearest(self._label, t)
        if lp is not None:
            for line in Path(lp).read_text().splitlines():
                q = line.split()
                if len(q) < 8:
                    continue
                cls = _PHENIKAA_CLASS.get(q[7])
                if cls is None:
                    continue
                x, y, z, l, w, h, yaw = map(float, q[:7])
                gt.append(dict(cls=cls, raw_cls=q[7], x=x, y=y, z=z, l=l, w=w, h=h, yaw=yaw))
        self.index += 1
        return Frame(t, images, {c: self.calib[c] for c in self.cameras}, ego_T, gt, i)

    def close(self):
        pass
