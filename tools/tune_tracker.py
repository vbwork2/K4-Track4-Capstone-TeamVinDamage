"""Tune tracker/Kalman parameters against Phenikaa LiDAR labels.

Observations (detection -> geometry -> fusion) do not depend on tracker settings, so they
are computed once per scene; each config only re-runs Tracker + zones.
Metrics per config (GT within 20 m, class groups person/rider/car):
  pos_med   median distance GT -> nearest same-class track (matched within 0.5 + 0.02 d^2 m)
  cover     fraction of GT with a matched track
  ghosts    tracks within 20 m per frame with no GT of the class nearby
  births    new track IDs per frame (fragmentation)
  m4_f1     SAT_SUON alert F1 per frame (same rule as src/eval_zones.py)
Usage: python -X utf8 tools/tune_tracker.py --scenes "COMPLEX CITY,COMPLEX URBAN" --model yolo11m
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cache import DetectionCache  # noqa: E402
from src.config import load_config  # noqa: E402
from src.eval_zones import footprint_in_side_zone  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.tracker import Tracker  # noqa: E402
from src.zones import evaluate_tracks  # noqa: E402

FIT = {"radial": [0.1, 0.009], "tangential": [0.05, 0.01]}  # from results/m2_phenikaa_*.json noise_by_range
R_GT = 20.0


class _Recorder:
    def __init__(self):
        self.calls = []

    def update(self, fused, t, ego_T):
        self.calls.append((list(fused), t, np.array(ego_T)))
        return []

    def reset(self):
        pass


def collect(root, scene, model, cfg):
    from src import geometry
    from src.sources import PhenikaaSource
    src = PhenikaaSource(root, scene)
    cache = DetectionCache(Path("cache") / "phenikaa" / scene / f"{model}.jsonl")
    pipe = Pipeline(cfg, geometry.ground_point, geometry.height_point, "phenikaa", scene, model)
    rec = pipe.tracker = _Recorder()
    gts = []
    for frame in src:
        pipe.process(frame, cache.read(frame))
        gts.append(frame.gt or [])
    return rec.calls, gts


def score(calls, gts, tcfg, cfg):
    tracker = Tracker(**tcfg)
    errs, n_gt, n_hit, ghosts, births, tp, fp, fn = [], 0, 0, 0, 0, 0, 0, 0
    last_id = 0
    for (fused, t, ego), gt in zip(calls, gts):
        tracks = evaluate_tracks(tracker.update(fused, t, ego), cfg["ego_box"], cfg["zone_cfg"], "phenikaa")
        births += sum(tr["id"] > last_id for tr in tracks)
        last_id = max([last_id] + [tr["id"] for tr in tracks])
        near_gt = [g for g in gt if math.hypot(g["x"], g["y"]) < R_GT]
        used = set()
        for g in near_gt:
            n_gt += 1
            d = math.hypot(g["x"], g["y"])
            radius = 0.5 + 0.02 * d * d
            best = min(((math.hypot(tr["x"]-g["x"], tr["y"]-g["y"]), i) for i, tr in enumerate(tracks)
                        if tr["cls"] == g["cls"] and i not in used), default=(1e9, -1))
            if best[0] <= radius:
                used.add(best[1]); n_hit += 1; errs.append(best[0])
        for i, tr in enumerate(tracks):
            d = math.hypot(tr["x"], tr["y"])
            if i in used or d >= R_GT:
                continue
            if not any(g["cls"] == tr["cls"] and math.hypot(tr["x"]-g["x"], tr["y"]-g["y"]) <= 0.5 + 0.02*d*d + 1
                       for g in gt):
                ghosts += 1
        occ = any(footprint_in_side_zone(g, cfg["ego_box"], cfg["zone_cfg"]) for g in gt)
        alert = any(tr["level"] > 0 and "SAT_SUON" in tr.get("zones", [tr["zone"]]) for tr in tracks)
        tp += occ and alert; fp += alert and not occ; fn += occ and not alert
    n = len(calls)
    f1 = 2*tp/(2*tp+fp+fn) if tp else 0.0
    return dict(pos_med=float(np.median(errs)) if errs else None, cover=n_hit/max(n_gt, 1),
                ghosts=ghosts/n, births=births/n, m4_p=tp/max(tp+fp, 1), m4_r=tp/max(tp+fn, 1), m4_f1=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/phenikaa")
    ap.add_argument("--scenes", default="COMPLEX CITY,COMPLEX URBAN")
    ap.add_argument("--model", default="yolo11m")
    ap.add_argument("--out", default="results/tune_tracker.json")
    a = ap.parse_args()
    cfg = load_config("phenikaa")
    data = {s: collect(a.root, s, a.model, cfg) for s in a.scenes.split(",")}
    base = dict(cfg["tracker"])
    for k in ("noise_model", "gate_chi2", "drop_tentative"):
        base.pop(k, None)
    configs = [("baseline (BE default)", dict(base))]
    for pv, chi, cap, drop, age in itertools.product([1.0, 4.0, 10.0], [9.21, 16.0], [3.0, 6.0],
                                                     [False, True], [0.5, 1.0]):
        configs.append((f"fit pv={pv} chi2={chi} cap={cap} drop={drop} age={age}",
                        dict(base, process_var=pv, gate_chi2=chi, gate_m=cap, drop_tentative=drop,
                             max_age_s=age, noise_model=FIT)))
    rows = []
    for name, tcfg in configs:
        per = {s: score(calls, gts, tcfg, cfg) for s, (calls, gts) in data.items()}
        mean = {k: float(np.mean([p[k] for p in per.values() if p[k] is not None])) for k in per[next(iter(per))]}
        rows.append(dict(name=name, tracker=tcfg, mean=mean, per_scene=per))
        print(f"{name:55s} pos {mean['pos_med']:.2f} cover {mean['cover']:.2f} ghosts {mean['ghosts']:.1f} "
              f"births {mean['births']:.1f} M4 F1 {mean['m4_f1']:.2f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
