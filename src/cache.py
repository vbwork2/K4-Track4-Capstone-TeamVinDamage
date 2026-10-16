"""Read Core's JSONL detections and fail clearly on incomplete cache files."""

import json
import math
from pathlib import Path


class DetectionCache:
    def __init__(self, path, timestamp_tolerance_s=0.001):
        self.path = Path(path)
        self.tolerance = timestamp_tolerance_s
        self.by_frame = {}
        self.by_cam = {}
        with self.path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    cam, timestamp, boxes = row["cam"], float(row["t"]), row["boxes"]
                    if not math.isfinite(timestamp) or not isinstance(boxes, list):
                        raise ValueError("Invalid timestamp or boxes")
                    if "frame" in row:
                        key = (int(row["frame"]), cam)
                        if key in self.by_frame:
                            raise ValueError(f"Duplicate frame/camera {key}")
                        self.by_frame[key] = row
                    self.by_cam.setdefault(cam, []).append((timestamp, boxes))
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid cache {self.path}, line {number}: {exc}") from exc
        for rows in self.by_cam.values():
            rows.sort(key=lambda item: item[0])
            if any(a[0] == b[0] for a, b in zip(rows, rows[1:])):
                raise ValueError(f"Duplicate camera timestamp in cache: {self.path}")

    def read(self, frame):
        from bisect import bisect_left
        result = {}
        for cam in frame.images:
            row = self.by_frame.get((frame.frame, cam))
            if row is not None:
                if abs(float(row["t"])-frame.t) > self.tolerance:
                    raise ValueError(f"Cache frame/timestamp mismatch: frame={frame.frame}, cam={cam}")
                result[cam] = [dict(box, cam=cam) for box in row["boxes"]]
                continue
            rows = self.by_cam.get(cam, [])
            index = bisect_left(rows, (frame.t,), key=None)
            nearby = rows[max(0, index-1):index+1]
            if not nearby:
                raise ValueError(f"Cache has no detections for frame={frame.frame}, cam={cam}")
            timestamp, boxes = min(nearby, key=lambda item: abs(item[0]-frame.t))
            if abs(timestamp-frame.t) > self.tolerance:
                raise ValueError(f"Cache timestamp mismatch: frame={frame.frame}, cam={cam}")
            result[cam] = [dict(box, cam=cam) for box in boxes]
        return result
