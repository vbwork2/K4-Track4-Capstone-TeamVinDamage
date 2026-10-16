"""Write detection cache JSONL + Colab timing (M3). See docs/BE.md 'Detection cache location'."""
import argparse
import json
import statistics
import time
from pathlib import Path


def make_source(dataset, root, scene):
    if dataset == "phenikaa":
        from .sources import PhenikaaSource
        return PhenikaaSource(root, scene)
    from .sources import NuScenesSource
    return NuScenesSource(root, scene)


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def run(source, detector, path, max_frames=None):
    """Write the cache; return per-frame (all cameras as one batch) inference ms."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    times, n = [], 0
    with path.open("w", encoding="utf-8") as out:
        for frame in source:
            if max_frames is not None and n >= max_frames:
                break
            cams = list(frame.images)
            _sync()
            t0 = time.perf_counter()
            results = detector.batch([frame.images[c] for c in cams])
            _sync()
            times.append((time.perf_counter() - t0) * 1000)
            for cam, boxes in zip(cams, results):
                rows = [{"cls": b["cls"], "conf": round(float(b["conf"]), 3),
                         "xyxy": [round(float(v), 1) for v in b["xyxy"]]} for b in boxes]
                out.write(json.dumps({"t": frame.t, "frame": frame.frame, "cam": cam, "boxes": rows}) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"{n} frames", flush=True)
    return times


def summarize(times, warmup=3):
    t = times[warmup:] or times
    if not t:
        return {"n_frames": 0}
    s = sorted(t)
    pick = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]
    mean = statistics.fmean(t)
    return {"n_frames": len(times), "n_timed": len(t), "ms_p50": round(pick(0.5), 2),
            "ms_p95": round(pick(0.95), 2), "ms_mean": round(mean, 2), "fps": round(1000 / mean, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["phenikaa", "nuscenes"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--batch", type=int, default=6)  # one frame = one batch of all cameras
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args(argv)
    from .detect import Detector
    stem = Path(a.model).stem
    det = Detector(a.model, conf=a.conf, imgsz=a.imgsz)
    src = make_source(a.dataset, a.root, a.scene)
    path = Path(a.out) / a.dataset / a.scene / f"{stem}.jsonl"
    times = run(src, det, path, a.max_frames)
    gpu = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    res = {"dataset": a.dataset, "scene": a.scene, "model": stem, "gpu": gpu, "imgsz": a.imgsz,
           "conf": a.conf, **summarize(times)}
    rp = Path("results") / f"m3_colab_{a.dataset}_{a.scene.replace(' ', '_')}_{stem}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res), "->", path)


if __name__ == "__main__":
    main()
