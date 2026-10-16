"""Find candidate frames for limitation cases L1-L4 -> results/cases.json.

python tools/find_cases.py --dataset nuscenes --root <dir> --scenes a,b --cache cache --model yolo11m
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache import DetectionCache  # noqa: E402
from src.config import load_config  # noqa: E402
from src.eval_bev import (NIGHT_SCENES, class_group, make_source, match_greedy,  # noqa: E402
                          project_gt, gt_visible_ok)
from src.fuse import iou, merge_riders  # noqa: E402

SIDE_CAMS = {"CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
             "CAM_P_FL", "CAM_P_FR", "CAM_P_L", "CAM_P_R"}
EDGE_CAMS = SIDE_CAMS
FRONT = {"CAM_FRONT", "CAM_P_F"}


def top(cases, n=5):
    return sorted(cases, key=lambda c: -c["score"])[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["phenikaa", "nuscenes"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--model", default="yolo11m")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    from src import geometry as geom
    cfg = load_config(a.dataset)
    mc, ri = cfg["detection"]["min_conf"], cfg["detection"]["rider_iou"]
    cases = {"L1": [], "L2": [], "L3": [], "L4": []}
    for scene in a.scenes.split(","):
        cache = DetectionCache(Path(a.cache)/a.dataset/scene/f"{a.model}.jsonl")
        night = scene in NIGHT_SCENES
        for frame in make_source(a.dataset, a.root, scene):
            boxes = cache.read(frame)
            pts = {}
            for cam, calib in frame.calib.items():
                dets = merge_riders(boxes.get(cam, []), mc, ri)
                h, w = frame.images[cam].shape[:2]
                tag = dict(scene=scene, frame=frame.frame, cam=cam)
                riders = [d for d in dets if d["cls"] == "rider"]
                # L1: >=3 riders whose boxes overlap one another
                groups = [[r for r in riders if r is r0 or iou(r["xyxy"], r0["xyxy"]) > 0.05] for r0 in riders]
                best = max(groups, key=len, default=[])
                if len(best) >= 3:
                    cases["L1"].append(dict(tag, score=len(best), reason=f"{len(best)} overlapping riders"))
                # L2: rider touching left/right/bottom edge in side cams
                if cam in EDGE_CAMS:
                    for r in riders:
                        x1, _, x2, y2 = r["xyxy"]
                        sides = [n for n, c in (("left", x1 <= 1), ("right", x2 >= w-1), ("bottom", y2 >= h-1)) if c]
                        if sides:
                            cases["L2"].append(dict(tag, score=r["conf"], reason=f"rider touches {'/'.join(sides)} edge"))
                # L3: night frames, GT riders/persons in front cam unmatched
                if night and cam in FRONT:
                    gts = []
                    for g in frame.gt or []:
                        if g["cls"] in ("person", "rider") and gt_visible_ok(g, a.dataset):
                            box = project_gt(g, calib, geom.project_ego)
                            if box:
                                gts.append(dict(g, xyxy=box))
                    matched = {j for _, j, _ in match_greedy(dets, gts)}
                    miss = len(gts)-len(matched)
                    if miss:
                        cases["L3"].append(dict(tag, score=miss, reason=f"night: {miss}/{len(gts)} GT person/rider missed"))
                # ground points for L4
                for d in dets:
                    hit = geom.ground_point(d.get("ground_xyxy", d["xyxy"]), calib)
                    if hit is not None and math.isfinite(hit.x):
                        pts.setdefault(cam, []).append((class_group(d["cls"]), hit.x, hit.y))
            # L4: same-class dets from different cameras within 3 m but >1.5 m apart
            cams = sorted(pts)
            for i, c1 in enumerate(cams):
                for c2 in cams[i+1:]:
                    for k1, x1, y1 in pts[c1]:
                        for k2, x2, y2 in pts[c2]:
                            d = math.hypot(x1-x2, y1-y2)
                            if k1 == k2 and 1.5 < d < 3.0:
                                cases["L4"].append(dict(scene=scene, frame=frame.frame, cam=f"{c1}+{c2}", score=-d,
                                                        reason=f"{k1} seen {d:.2f} m apart by two cameras"))
    result = {k: top(v) for k, v in cases.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out/"cases.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print({k: len(v) for k, v in cases.items()}, "->", out/"cases.json")


if __name__ == "__main__":
    main()
