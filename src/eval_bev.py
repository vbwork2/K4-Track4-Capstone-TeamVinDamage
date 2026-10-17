"""M2: BEV localisation error of detections vs 3D GT (python -m src.eval_bev --help)."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .fuse import iou, merge_riders

FRONT = {"CAM_FRONT", "CAM_P_F"}
BACK = {"CAM_BACK", "CAM_P_B"}
BINS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, math.inf)]
NIGHT_SCENES = {"scene-1094"}


def cam_group(cam):
    return "front" if cam in FRONT else "back" if cam in BACK else "side"


def class_group(cls):
    return "rider" if cls in ("rider", "motorcycle", "bicycle") else \
        "car" if cls in ("car", "bus", "truck") else cls


def bin_label(d):
    for lo, hi in BINS:
        if lo <= d < hi:
            return f"{lo}-{'inf' if hi == math.inf else hi}"
    return "n/a"


def gt_corners(g):
    """8 corners (8x3) of a GT box in ego frame (x, y, z centre; l along yaw)."""
    c, s = math.cos(g["yaw"]), math.sin(g["yaw"])
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                lx, ly = sx*g["l"]/2, sy*g["w"]/2
                pts.append([g["x"]+c*lx-s*ly, g["y"]+s*lx+c*ly, g["z"]+sz*g["h"]/2])
    return np.array(pts)


def project_gt(g, cam, project_ego, min_inside=0.5):
    """2D box of a GT box, or None if behind camera / <50% inside the image."""
    px, valid = project_ego(gt_corners(g), cam)
    px = np.asarray(px)
    if not np.all(valid):
        return None
    x1, y1 = px.min(0)
    x2, y2 = px.max(0)
    w, h = cam.size
    area = (x2-x1)*(y2-y1)
    if area <= 0:
        return None
    inter = max(0, min(x2, w)-max(x1, 0)) * max(0, min(y2, h)-max(y1, 0))
    if inter/area < min_inside:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def match_greedy(dets, gts, thr=0.5):
    """dets/gts: dicts with 'xyxy' and 'cls'. Greedy by IoU, same class group.
    Returns list of (det_idx, gt_idx, iou)."""
    cand = []
    for i, d in enumerate(dets):
        for j, g in enumerate(gts):
            if class_group(d["cls"]) != class_group(g["cls"]):
                continue
            v = iou(d["xyxy"], g["xyxy"])
            if v >= thr:
                cand.append((v, i, j))
    used_d, used_g, out = set(), set(), []
    for v, i, j in sorted(cand, reverse=True):
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        out.append((i, j, v))
    return out


def stats(errs):
    if not errs:
        return dict(n=0, mean=None, median=None, p90=None)
    a = np.asarray(errs)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), p90=float(np.percentile(a, 90)))


def noise_by_range(rows, edges=(0, 5, 10, 15, 20, 25, 30)):
    """Robust std (1.4826*MAD) of radial and tangential residuals per range bin, for the tracker noise model."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        sel = [r for r in rows if lo <= r["dist"] < hi]
        if len(sel) < 10:
            continue
        rad = np.array([r["rad"] for r in sel]); tan = np.array([r["tan"] for r in sel])
        mad = lambda a: float(1.4826*np.median(np.abs(a-np.median(a))))
        out.append(dict(range=[lo, hi], n=len(sel), d_mid=(lo+hi)/2, rad_bias=float(np.median(rad)),
                        rad_std=mad(rad), tan_bias=float(np.median(tan)), tan_std=mad(tan)))
    return out


def kalman_R(res, max_dist=30.0):
    r = np.array([(dx, dy) for dx, dy, d in res if d <= max_dist])
    if len(r) < 3:
        return None
    return dict(n=len(r), mean=r.mean(0).tolist(), cov=np.cov(r.T).tolist(), std=r.std(0).tolist())


def make_source(dataset, root, scene):
    from . import sources
    if dataset == "nuscenes":
        return sources.NuScenesSource(root, scene)
    return sources.PhenikaaSource(root, scene)


def gt_visible_ok(g, dataset):
    return dataset != "nuscenes" or g.get("visibility", 4) >= 2


def locate(det, cam, cfg, geom, image_shape):
    """Same ground/height logic as the pipeline; returns Hit or None."""
    h = image_shape[0]
    point_box = det.get("ground_xyxy", det["xyxy"])
    if point_box[3] >= h-1:
        return geom.height_point(det["xyxy"], cam, cfg["height_m"][det["cls"]])
    return geom.ground_point(point_box, cam)


def evaluate(dataset, root, scenes, cache_dir, model, out, source_factory=None, geom=None):
    from .cache import DetectionCache
    from .config import load_config
    if geom is None:
        from . import geometry as geom
    cfg = load_config(dataset)
    mc, ri = cfg["detection"]["min_conf"], cfg["detection"]["rider_iou"]
    rows = []
    kept_gt = matched_gt = 0
    for scene in scenes:
        src = (source_factory or make_source)(dataset, root, scene)
        cache = DetectionCache(Path(cache_dir)/dataset/scene/f"{model}.jsonl")
        for frame in src:
            boxes = cache.read(frame)
            for cam, calib in frame.calib.items():
                dets = merge_riders(boxes.get(cam, []), mc, ri)
                gts = []
                for g in frame.gt or []:
                    if not gt_visible_ok(g, dataset):
                        continue
                    box = project_gt(g, calib, geom.project_ego)
                    if box is not None:
                        gts.append(dict(g, xyxy=box))
                kept_gt += len(gts)
                for i, j, _ in match_greedy(dets, gts):
                    matched_gt += 1
                    hit = locate(dets[i], calib, cfg, geom, frame.images[cam].shape)
                    if hit is None or not (math.isfinite(hit.x) and math.isfinite(hit.y)):
                        continue
                    g = gts[j]
                    dx, dy = hit.x-g["x"], hit.y-g["y"]
                    dist = math.hypot(g["x"], g["y"])
                    ux, uy = g["x"]/max(dist, 1e-6), g["y"]/max(dist, 1e-6)
                    rows.append(dict(err=math.hypot(dx, dy), dx=dx, dy=dy, dist=dist,
                                     rad=dx*ux+dy*uy, tan=-dx*uy+dy*ux,
                                     cam=cam, cls=class_group(g["cls"]), scene=scene))
    return summarize(rows, kept_gt, matched_gt, dataset, model, scenes, out)


def summarize(rows, kept_gt, matched_gt, dataset, model, scenes, out=None):
    def group(keyf):
        d = {}
        for r in rows:
            d.setdefault(keyf(r), []).append(r["err"])
        return {k: stats(v) for k, v in sorted(d.items())}
    result = dict(dataset=dataset, model=model, scenes=list(scenes), n_pairs=len(rows),
                  overall=stats([r["err"] for r in rows]),
                  by_distance=group(lambda r: bin_label(r["dist"])),
                  by_camera_group=group(lambda r: cam_group(r["cam"])),
                  by_camera=group(lambda r: r["cam"]),
                  by_class=group(lambda r: r["cls"]),
                  recall=dict(matched_gt=matched_gt, kept_gt=kept_gt,
                              recall=matched_gt/kept_gt if kept_gt else None),
                  kalman_R=kalman_R([(r["dx"], r["dy"], r["dist"]) for r in rows]),
                  noise_by_range=noise_by_range(rows) if rows and "rad" in rows[0] else [])
    if dataset == "nuscenes":
        result["by_day_night"] = group(lambda r: "night" if r["scene"] in NIGHT_SCENES else "day")
    if out:
        p = Path(out)
        p.mkdir(parents=True, exist_ok=True)
        with (p/f"m2_{dataset}_{model}.json").open("w", encoding="utf-8") as f:
            json.dump(dict(result, residuals=[[r["dx"], r["dy"]] for r in rows]), f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["phenikaa", "nuscenes"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--model", default="yolo11m")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    res = evaluate(a.dataset, a.root, a.scenes.split(","), a.cache, a.model, a.out)
    print(json.dumps({k: v for k, v in res.items() if k != "by_camera"}, indent=2))


if __name__ == "__main__":
    main()
