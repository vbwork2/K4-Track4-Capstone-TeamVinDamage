"""Backend orchestration with injectable Core and FE dependencies."""

import time
from collections import deque

from .contracts import Observation
from .fuse import fuse_observations, merge_riders
from .tracker import Tracker
from .zones import alert_counts, evaluate_tracks


class Pipeline:
    def __init__(self, config, ground_point, height_point, source, scene="", model="yolo11n"):
        self.config = config
        self.ground_point = ground_point
        self.height_point = height_point
        self.source = source
        self.scene = scene
        self.model = model
        self.tracker = Tracker(**config["tracker"])
        self.durations = deque(maxlen=30)

    def reset(self):
        self.tracker.reset()
        self.durations.clear()

    def process(self, frame, boxes_by_cam, started=None, estimated_calib=False):
        started = time.perf_counter() if started is None else started
        detections, observations = [], []
        cfg = self.config
        for cam, image in frame.images.items():
            boxes = [dict(box, cam=cam) for box in boxes_by_cam.get(cam, [])]
            for det in merge_riders(boxes, cfg["detection"]["min_conf"], cfg["detection"]["rider_iou"]):
                det.update(track_id=None, level=0, uncertain=False)
                calibration = frame.calib[cam]
                point_box = det.get("ground_xyxy", det["xyxy"])
                height, width = image.shape[:2]
                visible_box = det["xyxy"]
                at_edge = (visible_box[0] <= 1 or visible_box[1] <= 1
                           or visible_box[2] >= width-1 or visible_box[3] >= height-1)
                bottom_edge = point_box[3] >= height-1
                H = cfg["height_m"][det["cls"]]
                if self.source == "webcam":
                    hit = self.height_point(det["xyxy"], calibration, H)
                elif bottom_edge:
                    hit = self.height_point(det["xyxy"], calibration, H)
                else:
                    hit = self.ground_point(point_box, calibration)
                reasons = []
                if det["conf"] < cfg["detection"]["uncertain_conf"]:
                    reasons.append("low_confidence")
                if at_edge or (hit is not None and hit.at_edge):
                    reasons.append("image_edge")
                if estimated_calib:
                    reasons.append("estimated_calibration")
                if hit is None:
                    reasons.append("invalid_geometry")
                else:
                    import math
                    if not all(math.isfinite(v) for v in (hit.x, hit.y, hit.dist)):
                        reasons.append("invalid_geometry")
                        hit = None
                    elif hit.dist > cfg["zone_cfg"]["max_range_m"]:
                        reasons.append("out_of_range")
                det["uncertain"] = bool(reasons)
                det["reasons"] = reasons
                detections.append(det)
                if hit is not None:
                    observations.append(Observation(det["cls"], det["conf"], hit.x, hit.y,
                                                    [det], bool(reasons), reasons))
        fused = fuse_observations(observations, **cfg["fusion"])
        tracks = evaluate_tracks(self.tracker.update(fused, frame.t, frame.ego_T),
                                 cfg["ego_box"], cfg["zone_cfg"], self.source)
        # Fusion copies detections; map by camera and box to update the FE list.
        annotations = {}
        for track in tracks:
            for det in track.pop("dets"):
                annotations[(det["cam"], tuple(det["xyxy"]))] = track
        for det in detections:
            track = annotations.get((det["cam"], tuple(det["xyxy"])))
            if track is not None:
                det.update(track_id=track["id"], level=track["level"],
                           uncertain=track["uncertain"], reasons=track["reasons"])
        ended = time.perf_counter()
        latency = (ended-frame.t)*1000 if self.source == "webcam" else (ended-started)*1000
        self.durations.append(ended-started)
        fps = len(self.durations) / max(sum(self.durations), 1e-9)
        return dict(t=frame.t, source=self.source, scene=self.scene, model=self.model,
                    images=frame.images, dets=detections, tracks=tracks, gt=frame.gt or [],
                    ego_box=cfg["ego_box"], zone_cfg=cfg["zone_cfg"],
                    stats=dict(fps=fps, latency_ms=latency, n_tracks=len(tracks),
                               alerts=alert_counts(tracks), estimated_calib=estimated_calib))
