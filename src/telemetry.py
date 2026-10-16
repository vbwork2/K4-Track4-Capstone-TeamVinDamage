"""Stream decision logs and summarize measured processing latency."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class DecisionLog:
    FIELDS = ("frame", "t", "source", "scene", "model", "track_id", "cls", "x", "y", "vx", "vy",
              "zone", "zones", "level", "ttc", "uncertain", "reasons", "fps", "latency_ms", "n_tracks",
              "frame_uncertain", "n_uncertain_dets", "SAT_SUON", "TAT_DAU", "PHIA_TRUOC")

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.stream, self.FIELDS)
        self.writer.writeheader()

    def write(self, frame, state):
        stats = state["stats"]
        base = dict(frame=frame, t=state["t"], source=state["source"], scene=state["scene"],
                    model=state["model"], fps=stats["fps"], latency_ms=stats["latency_ms"],
                    n_tracks=stats["n_tracks"], **stats["alerts"])
        base["n_uncertain_dets"] = sum(det["uncertain"] for det in state["dets"])
        base["frame_uncertain"] = bool(base["n_uncertain_dets"] or any(t["uncertain"] for t in state["tracks"]))
        for track in state["tracks"] or [None]:
            row = dict(base)
            if track:
                row.update({key: track.get(key) for key in
                            ("cls", "x", "y", "vx", "vy", "zone", "level", "ttc", "uncertain")})
                row.update(track_id=track["id"], zones="|".join(track.get("zones", [])),
                           reasons="|".join(track.get("reasons", [])))
            self.writer.writerow(row)
        self.stream.flush()

    def close(self):
        self.stream.close()


class TimingSummary:
    def __init__(self):
        self.latencies = []
        self.frame_seconds = []
        self.by_model = {}

    def add(self, latency_ms, frame_seconds, model="unknown"):
        self.latencies.append(latency_ms)
        self.frame_seconds.append(frame_seconds)
        group = self.by_model.setdefault(model, {"latency_ms": [], "frame_seconds": []})
        group["latency_ms"].append(latency_ms)
        group["frame_seconds"].append(frame_seconds)

    def save(self, path, source, model, synthetic=False):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        result = dict(source=source, model=model, n_frames=len(self.latencies), synthetic=synthetic,
                      recorded_at=datetime.now(timezone.utc).isoformat(),
                      scope="capture_to_decision" if source == "webcam" else "local_replay_processing",
                      fps_scope="capture_read_to_display_including_recording_excluding_pause_and_replay_wait",
                      fps=None, latency_p50_ms=None, latency_p95_ms=None)
        if self.latencies:
            result.update(fps=len(self.frame_seconds)/max(sum(self.frame_seconds), 1e-9),
                          latency_p50_ms=float(np.percentile(self.latencies, 50)),
                          latency_p95_ms=float(np.percentile(self.latencies, 95)))
        result["by_model"] = {
            name: dict(n_frames=len(group["latency_ms"]),
                       fps=len(group["frame_seconds"])/max(sum(group["frame_seconds"]), 1e-9),
                       latency_p50_ms=float(np.percentile(group["latency_ms"], 50)),
                       latency_p95_ms=float(np.percentile(group["latency_ms"], 95)))
            for name, group in self.by_model.items()
        }
        if len(self.by_model) > 1:
            result["model"] = "mixed"
        with target.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
