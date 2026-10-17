"""M4: side-zone alert precision/recall vs GT footprints."""

import argparse
import csv
import json
import math
from pathlib import Path

from .zones import in_side_zone


def footprint_points(g):
    """4 corners + centre + 4 edge midpoints of the l x w rotated rectangle.
    Point sampling instead of exact polygon clipping: good enough for a 1 m wide zone."""
    c, s = math.cos(g["yaw"]), math.sin(g["yaw"])
    out = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            lx, ly = a*g["l"]/2, b*g["w"]/2
            out.append((g["x"]+c*lx-s*ly, g["y"]+s*lx+c*ly))
    return out


def footprint_in_side_zone(g, ego_box, zone_cfg):
    return any(in_side_zone(x, y, ego_box, zone_cfg) for x, y in footprint_points(g))


def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else None
    r = tp/(tp+fn) if tp+fn else None
    f = 2*p*r/(p+r) if p and r else (0.0 if p is not None and r is not None else None)
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f)


def score(pairs):
    """pairs: iterable of (gt_occupied, alerted)."""
    pairs = list(pairs)
    tp = sum(g and a for g, a in pairs)
    fp = sum((not g) and a for g, a in pairs)
    fn = sum(g and not a for g, a in pairs)
    return prf(tp, fp, fn)


def read_logs(paths):
    """-> {(scene, frame): dict(alert, uncertain)} deduped by frame."""
    out = {}
    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["scene"], int(row["frame"]))
                e = out.setdefault(key, dict(alert=False, uncertain=False))
                if "SAT_SUON" in (row.get("zones") or "").split("|") and int(float(row.get("level") or 0)) > 0:
                    e["alert"] = True
                if (row.get("frame_uncertain") or "").strip().lower() in ("true", "1"):
                    e["uncertain"] = True
    return out


def evaluate(dataset, root, scenes, logs, out, model=None, source_factory=None):
    from .config import load_config
    from .eval_bev import make_source
    cfg = load_config(dataset)
    log = read_logs(logs)
    all_pairs, cert_pairs = [], []
    for scene in scenes:
        for frame in (source_factory or make_source)(dataset, root, scene):
            e = log.get((scene, frame.frame))
            if e is None:
                continue
            occ = any(footprint_in_side_zone(g, cfg["ego_box"], cfg["zone_cfg"]) for g in frame.gt or [])
            all_pairs.append((occ, e["alert"]))
            if not e["uncertain"]:
                cert_pairs.append((occ, e["alert"]))
    result = dict(dataset=dataset, model=model, scenes=scenes, n_frames=len(all_pairs),
                  all_frames=score(all_pairs),
                  excluding_uncertain=dict(n_frames=len(cert_pairs), **score(cert_pairs)))
    if out:
        p = Path(out)
        p.mkdir(parents=True, exist_ok=True)
        with (p/f"m4_{dataset}_{model}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["phenikaa", "nuscenes"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--log", action="append", required=True)
    ap.add_argument("--model", default="yolo11m")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    print(json.dumps(evaluate(a.dataset, a.root, a.scenes.split(","), a.log, a.out, a.model), indent=2))


if __name__ == "__main__":
    main()
