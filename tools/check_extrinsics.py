"""Draw projected GT 3D boxes on CAM_P_F/L/B of one frame -> results/check_extrinsics.jpg."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import project_ego  # noqa: E402
from src.sources import PhenikaaSource  # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def corners(g):
    l, w, h = g["l"], g["w"], g["h"]
    c = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sz in (-1, 1)
                  for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    cy, sy_ = np.cos(g["yaw"]), np.sin(g["yaw"])
    R = np.array([[cy, -sy_, 0], [sy_, cy, 0], [0, 0, 1]])
    return c @ R.T + [g["x"], g["y"], g["z"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/phenikaa")
    ap.add_argument("--scene", default="COMPLEX CITY")
    ap.add_argument("--frame", type=int, default=30)
    ap.add_argument("--out", default="results/check_extrinsics.jpg")
    a = ap.parse_args()
    src = PhenikaaSource(a.root, a.scene)
    src.seek(a.frame)
    fr = next(src)
    tiles = []
    for cam in ("CAM_P_F", "CAM_P_L", "CAM_P_B"):
        img, cc = fr.images[cam].copy(), fr.calib[cam]
        for g in fr.gt:
            px, ok = project_ego(corners(g), cc)
            if ok.sum() < 4:
                continue
            for i, j in EDGES:
                if ok[i] and ok[j]:
                    cv2.line(img, tuple(px[i].astype(int)), tuple(px[j].astype(int)), (0, 255, 0), 3)
        cv2.putText(img, cam, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        tiles.append(cv2.resize(img, (640, 512)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.hstack(tiles))
    print("saved", a.out, "gt", len(fr.gt))


if __name__ == "__main__":
    main()
