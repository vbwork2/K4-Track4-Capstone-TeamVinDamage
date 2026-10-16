"""Replay and Live entry point for the backend pipeline."""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from .cache import DetectionCache
from .config import load_config
from .pipeline import Pipeline
from .telemetry import DecisionLog, TimingSummary


def _callable(reference):
    module, name = reference.split(":", 1)
    return getattr(importlib.import_module(module), name)


def _geometry(args):
    if args.source == "synthetic":
        from .synthetic import stub_ground_point, stub_height_point
        return stub_ground_point, stub_height_point
    try:
        module = importlib.import_module("src.geometry")
    except ModuleNotFoundError as exc:
        if exc.name != "src.geometry":
            raise
        raise RuntimeError("Core's src/geometry.py is missing. Use --source synthetic --headless for a BE-only check.") from exc
    return module.ground_point, module.height_point


def _source(args):
    if args.source == "synthetic":
        from .synthetic import SyntheticSource
        return SyntheticSource(args.max_frames or 60)
    if args.source == "webcam":
        from .sources import WebcamSource
        return WebcamSource(args.device, args.webcam_calib)
    if not args.root:
        raise ValueError("--root is required for a dataset source")
    if args.source == "nuscenes":
        from .sources import NuScenesSource
        return NuScenesSource(args.root, args.scene, args.version)
    module = importlib.import_module("src.sources")
    if not hasattr(module, "PhenikaaSource"):
        raise RuntimeError("Core must add PhenikaaSource(root, scene) to src/sources.py")
    return module.PhenikaaSource(args.root, args.scene)


def _cache(args, scene, model):
    if args.source in ("webcam", "synthetic"):
        return None
    path = Path(args.cache)
    if path.suffix != ".jsonl":
        path = path / args.source / scene / f"{Path(model).stem}.jsonl"
    return DetectionCache(path)


class _LiveDetector:
    """Thin Ultralytics adapter; Core may replace it via --detector."""

    def __init__(self, model, device=None):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.device = device

    def __call__(self, image):
        result = self.model.predict(image, conf=0.4, classes=[0, 1, 2, 3, 5, 7],
                                    device=self.device, verbose=False)[0]
        if result.boxes is None:
            return []
        return [dict(cls=int(cls), conf=float(conf), xyxy=list(map(float, xyxy)))
                for xyxy, conf, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                           result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]


class _Recording:
    def __init__(self, path, fps):
        self.path = Path(path)
        self.fps = fps
        self.writer = None
        self.size = None

    def write(self, image):
        import cv2
        height, width = image.shape[:2]
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.size = (width, height)
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot create recording: {self.path}")
        if self.size != (width, height):
            raise ValueError("FE render dimensions changed during recording")
        self.writer.write(image)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("phenikaa", "nuscenes", "webcam", "synthetic"), required=True)
    parser.add_argument("--root")
    parser.add_argument("--scene")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--config", help="JSON overrides, including measured tracker.measurement_var")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index")
    parser.add_argument("--inference-device", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--webcam-calib", default="calib/webcam.json")
    parser.add_argument("--detector", help="Optional module:callable accepting one BGR image and returning boxes")
    parser.add_argument("--renderer", default="src.ui:render")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--log", default="results/log.csv")
    parser.add_argument("--metrics", default="results/m3_local.json")
    parser.add_argument("--record", help="Write FE rendered frames to MP4")
    parser.add_argument("--record-fps", type=float, default=10)
    parser.add_argument("--replay-fps", type=float, default=0, help="0 follows dataset timestamps")
    return parser


def run(args):
    if args.max_frames < 0 or args.record_fps <= 0 or args.replay_fps < 0:
        raise ValueError("Frame limits and replay FPS must be nonnegative; recording FPS must be positive")
    if args.headless and args.record:
        raise ValueError("--record requires the FE renderer; remove --headless")
    cfg = load_config(args.source, args.config)
    ground, height = _geometry(args)
    renderer = None if args.headless else _callable(args.renderer)
    detector = None
    if args.source == "webcam":
        detector = _callable(args.detector) if args.detector else _LiveDetector(args.model, args.inference_device)
    source, log, recording = None, None, None
    timing = TimingSummary()
    processed = 0
    model = args.model
    opts = {"show_gt": False, "record": bool(args.record), "paused": False}
    try:
        source = _source(args)
        scene = getattr(source, "scene", args.scene or "")
        cache = _cache(args, scene, model)
        pipeline = Pipeline(cfg, ground, height, args.source, scene, Path(model).stem)
        log = DecisionLog(args.log)
        recording_path = args.record or "results/recording.mp4"
        recording = _Recording(recording_path, args.record_fps)

        def read_boxes(frame):
            if args.source == "synthetic":
                from .synthetic import synthetic_boxes
                return synthetic_boxes(frame)
            if cache is not None:
                return cache.read(frame)
            return {cam: detector(image) for cam, image in frame.images.items()}

        def process(frame, started):
            return pipeline.process(frame, read_boxes(frame), started,
                                    getattr(source, "estimated_calib", False))

        def rebuild(target):
            # Replaying history preserves velocity and IDs after a seek or model switch.
            source.seek(0)
            pipeline.reset()
            for _ in range(target+1):
                started = time.perf_counter()
                candidate = next(source)
                candidate_state = process(candidate, started)
            return candidate, candidate_state

        frame, state = None, None
        while True:
            cycle_start = time.perf_counter()
            new_frame = not opts["paused"] or frame is None
            if new_frame:
                try:
                    frame = next(source)
                except StopIteration:
                    break
                state = process(frame, cycle_start)
            if renderer:
                import cv2
                canvas = renderer(state, opts)
                cv2.imshow("Surround Rider Warning", canvas)
                if opts["record"] and new_frame:
                    recording.write(canvas)
            if new_frame:
                elapsed = time.perf_counter()-cycle_start
                state["stats"]["fps"] = 1/max(elapsed, 1e-9)
                timing.add(state["stats"]["latency_ms"], elapsed, Path(model).stem)
                log.write(frame.frame, state)
                processed += 1
                if args.headless and (processed == 1 or processed % 30 == 0):
                    print(json.dumps(dict(frame=frame.frame, synthetic=args.source == "synthetic", **state["stats"])))
                if args.max_frames and processed >= args.max_frames:
                    break
            if not renderer:
                continue
            # FE owns rendering; the application routes controls to source and model state.
            delay = 1 if args.source == "webcam" else 20
            if new_frame and args.source != "webcam":
                period = 1/args.replay_fps if args.replay_fps else getattr(source, "frame_period_s", None)
                if period is None:
                    period = 0.5 if args.source == "nuscenes" else 0.1
                delay = max(1, round((period-(time.perf_counter()-cycle_start))*1000))
            key = cv2.waitKeyEx(delay)
            if key in (ord("q"), 27) or cv2.getWindowProperty("Surround Rider Warning", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("g"):
                opts["show_gt"] = not opts["show_gt"]
            elif key == ord("r"):
                opts["record"] = not opts["record"]
                if not opts["record"]:
                    recording.close()
                    recording.path = recording.path.with_name(f"{recording.path.stem}_{time.time_ns()}.mp4")
            elif key == ord(" ") and args.source != "webcam":
                opts["paused"] = not opts["paused"]
            elif key in (2424832, 2555904, 81, 83) and hasattr(source, "seek"):
                direction = -1 if key in (2424832, 81) else 1
                target = max(0, min(len(source)-1, frame.frame+direction))
                frame, state = rebuild(target)
                opts["paused"] = True
            elif key == ord("m") and cache is not None and hasattr(source, "seek"):
                alternate = "yolo11m.pt" if Path(model).stem == "yolo11n" else "yolo11n.pt"
                try:
                    next_cache = _cache(args, scene, alternate)
                except FileNotFoundError:
                    print(f"Alternate cache missing: {alternate}", file=sys.stderr)
                    continue
                if Path(args.cache).suffix == ".jsonl":
                    print("Model switching requires a cache directory, not a single JSONL", file=sys.stderr)
                    continue
                cache, model = next_cache, alternate
                pipeline.model = Path(model).stem
                frame, state = rebuild(frame.frame)
        print(f"Processed {processed} frames; log: {args.log}; metrics: {args.metrics}")
        return 0
    finally:
        if recording:
            recording.close()
        if log:
            log.close()
        if source is not None and hasattr(source, "close"):
            source.close()
        if timing.latencies:
            timing.save(args.metrics, args.source, Path(model).stem, args.source == "synthetic")
        if renderer:
            import cv2
            cv2.destroyAllWindows()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError, AttributeError) as exc:
        print(f"Backend error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
