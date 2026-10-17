import unittest
from pathlib import Path

import numpy as np

from src.calib import load_phenikaa_calib
from src.contracts import CamCalib
from src.geometry import ground_point, project_ego

ROOT = Path(__file__).resolve().parents[1] / "data" / "phenikaa"


def synthetic_cam():
    # camera at (0,0,1.5) looking forward: cam z -> ego x, cam x -> -y, cam y -> -z
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [0, 0, 1.5]
    return CamCalib([[1000, 0, 960], [0, 1000, 768], [0, 0, 1]], [-0.3, 0.1, 0, 0, 0],
                    (1920, 1536), T, 0.0)


class GeometryTest(unittest.TestCase):
    def test_synthetic_roundtrip(self):
        cam = synthetic_cam()
        for p in [(10, 0, 0), (8, 2, 0), (15, -3, 0), (6, 1, 0)]:
            px, ok = project_ego(np.array([p], float), cam)
            self.assertTrue(ok[0])
            u, v = px[0]
            hit = ground_point([u - 20, v - 80, u + 20, v], cam)
            self.assertLess(np.hypot(hit.x - p[0], hit.y - p[1]), 0.05)

    def test_sky_ray_is_none(self):
        self.assertIsNone(ground_point([900, 100, 1000, 200], synthetic_cam()))

    def test_real_data(self):
        calib_dir = ROOT / "Calib Information"
        if not calib_dir.exists():
            self.skipTest("data/phenikaa missing")
        cam = load_phenikaa_calib(calib_dir, -0.04)["CAM_P_F"]
        errs = []
        for f in sorted((ROOT / "COMPLEX CITY" / "label").glob("*.txt"))[::5]:
            if "(" in f.name:
                continue
            for line in f.read_text().splitlines():
                q = line.split()
                if len(q) < 8:
                    continue
                x, y = float(q[0]), float(q[1])
                if not (3 < x < 25 and abs(y) < 15):
                    continue
                px, ok = project_ego(np.array([[x, y, -0.04]]), cam)
                if not ok[0] or px[0, 1] > cam.size[1] - 3:
                    continue
                u, v = px[0]
                hit = ground_point([u - 10, v - 50, u + 10, v], cam)
                if hit:
                    errs.append(np.hypot(hit.x - x, hit.y - y))
        self.assertGreater(len(errs), 10)
        med = float(np.median(errs))
        print(f"real-data median error {med:.4f} m over {len(errs)} labels")
        self.assertLess(med, 0.3)


if __name__ == "__main__":
    unittest.main()
