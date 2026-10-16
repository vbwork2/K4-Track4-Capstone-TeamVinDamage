"""Deterministic backend checks without dataset, webcam, OpenCV, or GPU."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.app import main
from src.cache import DetectionCache
from src.config import load_config
from src.contracts import CamCalib, Frame, Hit, Observation
from src.fuse import fuse_observations, merge_riders
from src.pipeline import Pipeline
from src.sources import NuScenesSource, load_webcam_calib, pose_matrix
from src.telemetry import DecisionLog, TimingSummary
from src.tracker import Tracker
from src.zones import alert_counts, evaluate_track


def observation(x, y=0, cam="CAM_FRONT", cls="car", **kwargs):
    return Observation(cls, 0.9, x, y, [dict(cam=cam, xyxy=[1, 2, 3, 4])], **kwargs)


class FusionTests(unittest.TestCase):
    def test_rider_pairing_preserves_cycle_ground_and_does_not_double_assign(self):
        boxes = [dict(cls="person", conf=.9, xyxy=[20, 10, 40, 80]),
                 dict(cls="person", conf=.8, xyxy=[25, 20, 45, 80]),
                 dict(cls="motorcycle", conf=.85, xyxy=[10, 60, 60, 100])]
        merged = merge_riders(boxes)
        self.assertEqual([b["cls"] for b in merged].count("rider"), 1)
        self.assertEqual([b["cls"] for b in merged].count("person"), 1)
        rider = next(b for b in merged if b["cls"] == "rider")
        self.assertEqual(rider["ground_xyxy"], boxes[2]["xyxy"])
        self.assertEqual(boxes[2]["cls"], "motorcycle")

    def test_rider_does_not_merge_across_cameras(self):
        boxes = [dict(cam="A", cls="person", conf=.9, xyxy=[20, 10, 40, 80]),
                 dict(cam="B", cls="motorcycle", conf=.9, xyxy=[10, 60, 60, 100])]
        self.assertEqual(len(merge_riders(boxes)), 2)

    def test_filter_invalid_and_low_confidence_boxes(self):
        boxes = [dict(cls=2, conf=.9, xyxy=[1, 1, 3, 4]),
                 dict(cls="car", conf=.1, xyxy=[1, 1, 3, 4]),
                 dict(cls="car", conf=.9, xyxy=[3, 1, 1, 4]),
                 dict(cls="car", conf=float("nan"), xyxy=[1, 1, 3, 4])]
        self.assertEqual(len(merge_riders(boxes)), 1)

    def test_cross_camera_fusion_preserves_provenance(self):
        original = [observation(5), observation(5.4, cam="CAM_LEFT")]
        result = fuse_observations(original)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].x, 5.2)
        self.assertEqual(len(result[0].dets), 2)
        self.assertEqual(len(original[0].dets), 1)

    def test_same_camera_and_chained_duplicates_stay_separate(self):
        self.assertEqual(len(fuse_observations([observation(5), observation(5.1)])), 2)
        result = fuse_observations([observation(5, cam="A"), observation(5.8, cam="B"), observation(6.6, cam="C")])
        self.assertEqual(len(result), 2)

    def test_camera_disagreement_is_uncertain_and_not_merged(self):
        result = fuse_observations([observation(5), observation(7, cam="CAM_LEFT")])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(o.uncertain and "camera_disagreement" in o.reasons for o in result))


class TrackerTests(unittest.TestCase):
    def test_tracks_across_camera_change_and_estimates_velocity(self):
        tracker = Tracker(measurement_var=.01)
        ids = []
        for frame in range(20):
            tracks = tracker.update([observation(10-frame*.1, cam="A" if frame < 10 else "B")], frame*.1, np.eye(4))
            ids.append(tracks[0]["id"])
        self.assertEqual(len(set(ids)), 1)
        self.assertAlmostEqual(tracks[0]["vx"], -1, delta=.15)
        self.assertTrue(tracks[0]["velocity_valid"])
        self.assertLessEqual(len(tracks[0]["trail"]), 11)

    def test_stationary_world_object_has_relative_closing_speed(self):
        tracker = Tracker(measurement_var=.01)
        for frame in range(10):
            ego = np.eye(4)
            ego[0, 3] = frame*.2
            tracks = tracker.update([observation(10-frame*.2)], frame*.1, ego)
        self.assertAlmostEqual(tracks[0]["vx"], -2, places=6)
        self.assertAlmostEqual(tracker.tracks[0].state[2], 0, places=6)
        self.assertEqual(tracks[0]["id"], 1)

    def test_ego_rotation_does_not_move_stationary_world_track(self):
        tracker = Tracker()
        tracker.update([observation(10)], 0, np.eye(4))
        angle = .1
        ego = pose_matrix(dict(rotation=[np.cos(angle/2), 0, 0, np.sin(angle/2)], translation=[0, 0, 0]))
        tracks = tracker.update([observation(10*np.cos(angle), -10*np.sin(angle))], .5, ego)
        self.assertEqual(tracks[0]["id"], 1)
        np.testing.assert_allclose(tracker.tracks[0].state[:2], [10, 0], atol=1e-8)

    def test_prediction_is_uncertain_and_expires(self):
        tracker = Tracker(max_age_s=.5)
        tracker.update([observation(5)], 0, np.eye(4))
        self.assertTrue(tracker.update([], .1, np.eye(4))[0]["uncertain"])
        self.assertEqual(tracker.update([], .6, np.eye(4)), [])

    def test_invalid_time_and_covariance_fail(self):
        with self.assertRaises(ValueError):
            Tracker(measurement_var=0)
        tracker = Tracker()
        tracker.update([], 0, np.eye(4))
        with self.assertRaises(ValueError):
            tracker.update([], 0, np.eye(4))
        tracker.reset()
        self.assertEqual(tracker.update([], 0, np.eye(4)), [])

    def test_kalman_outlier_sets_uncertainty(self):
        tracker = Tracker(measurement_var=.0001, initial_velocity_var=.0001, process_var=0)
        tracker.update([observation(5)], 0, np.eye(4))
        track = tracker.update([observation(6)], .1, np.eye(4))[0]
        self.assertTrue(track["uncertain"])
        self.assertIn("kalman_residual", track["reasons"])


class ZoneTests(unittest.TestCase):
    def setUp(self):
        cfg = load_config("nuscenes")
        self.ego, self.cfg = cfg["ego_box"], cfg["zone_cfg"]

    def evaluate(self, source="nuscenes", **updates):
        track = dict(id=1, cls="car", x=5, y=0, vx=-5, vy=0, uncertain=False)
        track.update(updates)
        return evaluate_track(track, self.ego, self.cfg, source)

    def test_front_ttc_and_stationary_receding_cases(self):
        self.assertEqual(self.evaluate()["level"], 2)
        self.assertEqual(self.evaluate(x=10)["level"], 1)
        self.assertEqual(self.evaluate(x=20)["level"], 0)
        self.assertIsNone(self.evaluate(vx=0)["ttc"])
        self.assertIsNone(self.evaluate(vx=2)["ttc"])
        self.assertIsNone(self.evaluate(x=-5)["ttc"])
        self.assertIsNone(self.evaluate(velocity_valid=False)["ttc"])

    def test_uncertainty_never_produces_red(self):
        self.assertEqual(self.evaluate(uncertain=True)["level"], 1)
        result = self.evaluate(x=31, vx=-40)
        self.assertEqual(result["level"], 1)
        self.assertIn("out_of_range", result["reasons"])

    def test_side_and_cut_in_and_live_restrictions(self):
        side = self.evaluate(x=0, y=1.5, vx=0)
        self.assertEqual(side["zone"], "SAT_SUON")
        self.assertEqual(alert_counts([side])["SAT_SUON"], 1)
        self.assertEqual(self.evaluate(x=5, y=2, vy=-1)["zone"], "TAT_DAU")
        self.assertIsNone(self.evaluate(x=5, y=2, vy=1)["zone"])
        self.assertIsNone(self.evaluate(source="webcam", x=0, y=1.5, vx=0)["zone"])
        self.assertIsNone(self.evaluate(source="webcam", x=5, y=2, vy=-1)["zone"])

    def test_nonfinite_state_never_alarms(self):
        result = self.evaluate(vx=float("nan"))
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["level"], 0)


class CacheAndPipelineTests(unittest.TestCase):
    def test_cache_matches_frame_and_timestamp_and_rejects_missing_camera(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"cache.jsonl"
            path.write_text(json.dumps(dict(t=1, frame=0, cam="A", boxes=[]))+"\n", encoding="utf-8")
            cache = DetectionCache(path)
            frame = Frame(1, {"A": None}, {}, np.eye(4), frame=0)
            self.assertEqual(cache.read(frame), {"A": []})
            frame.t = 1.1
            with self.assertRaises(ValueError):
                cache.read(frame)
            frame.t = 1
            frame.frame = 100
            self.assertEqual(cache.read(frame), {"A": []})
            frame.t = 2
            with self.assertRaises(ValueError):
                cache.read(frame)
            frame.images["B"] = None
            with self.assertRaises(ValueError):
                cache.read(frame)

    def test_duplicate_cache_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"cache.jsonl"
            row = json.dumps(dict(t=1, frame=0, cam="A", boxes=[]))+"\n"
            path.write_text(row*2, encoding="utf-8")
            with self.assertRaises(ValueError):
                DetectionCache(path)

    def test_pipeline_propagates_track_warning_to_every_camera(self):
        config = load_config("nuscenes")
        def ground(box, cam):
            return Hit(0, 1.4, 1.4, "ground")
        pipe = Pipeline(config, ground, None, "nuscenes")
        image = np.zeros((200, 200, 3), np.uint8)
        calib = CamCalib(np.eye(3), [], (200, 200), np.eye(4))
        frame = Frame(1, {"A": image, "B": image}, {"A": calib, "B": calib}, np.eye(4))
        box = dict(cls="car", conf=.9, xyxy=[20, 20, 80, 180])
        result = pipe.process(frame, {"A": [box], "B": [box]})
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual({d["track_id"] for d in result["dets"]}, {1})
        self.assertTrue(all(d["level"] == 1 for d in result["dets"]))
        self.assertEqual(result["stats"]["alerts"]["SAT_SUON"], 1)

    def test_edge_box_uses_height_and_missing_ground_stays_uncertain(self):
        config = load_config("nuscenes")
        calls = []
        def height(box, cam, H):
            calls.append(H)
            return Hit(5, 0, 5, "height", True)
        pipe = Pipeline(config, lambda box, cam: None, height, "nuscenes")
        image = np.zeros((200, 200, 3), np.uint8)
        calib = CamCalib(np.eye(3), [], (200, 200), np.eye(4))
        frame = Frame(1, {"A": image}, {"A": calib}, np.eye(4))
        result = pipe.process(frame, {"A": [dict(cls="person", conf=.9, xyxy=[20, 20, 80, 200])]})
        self.assertEqual(calls, [1.7])
        self.assertTrue(result["tracks"][0]["uncertain"])
        pipe.reset()
        result = pipe.process(frame, {"A": [dict(cls="car", conf=.9, xyxy=[20, 20, 80, 180])]})
        self.assertEqual(result["tracks"], [])
        self.assertIn("invalid_geometry", result["dets"][0]["reasons"])

    def test_synthetic_cli_runs_to_csv_and_explicit_synthetic_metrics(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            log = Path(temp)/"log.csv"
            metrics = Path(temp)/"metrics.json"
            status = main(["--source", "synthetic", "--headless", "--max-frames", "60",
                           "--log", str(log), "--metrics", str(metrics)])
            self.assertEqual(status, 0)
            result = json.loads(metrics.read_text(encoding="utf-8"))
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["n_frames"], 60)
            with log.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 60)
            self.assertEqual({row["track_id"] for row in rows}, {"1"})
            self.assertTrue(all(row["uncertain"] == "True" for row in rows))
            self.assertTrue(any(row["zone"] == "PHIA_TRUOC" for row in rows))

    def test_config_rejects_inverted_thresholds_and_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"config.json"
            path.write_text(json.dumps({"zone_cfg": {"ttc_danger_s": 4}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config("webcam", path)
        first = load_config("webcam")
        first["ego_box"]["length"] = 100
        self.assertEqual(load_config("webcam")["ego_box"]["length"], 4.5)

    def test_metrics_keep_models_separate(self):
        timing = TimingSummary()
        timing.add(10, .1, "yolo11n")
        timing.add(100, .2, "yolo11m")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"metrics.json"
            timing.save(path, "nuscenes", "yolo11m")
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(result["model"], "mixed")
            self.assertEqual(result["by_model"]["yolo11n"]["latency_p50_ms"], 10)
            self.assertEqual(result["by_model"]["yolo11m"]["latency_p50_ms"], 100)

    def test_live_cli_with_fake_capture_and_injected_core(self):
        image = np.zeros((200, 200, 3), np.uint8)
        capture = SimpleNamespace(isOpened=lambda: True, read=lambda: (True, image), release=lambda: released.append(True))
        cv2 = SimpleNamespace(VideoCapture=lambda device: capture)
        released = []
        detector = lambda image: [dict(cls="person", conf=.9, xyxy=[20, 20, 80, 180])]
        geometry = lambda box, cam, H: Hit(2, 0, 2, "height")
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"cv2": cv2}), \
                patch("src.app._geometry", return_value=(None, geometry)), \
                patch("src.app._callable", return_value=detector), \
                self.assertWarns(UserWarning), redirect_stdout(io.StringIO()):
            status = main(["--source", "webcam", "--headless", "--max-frames", "3", "--detector", "fake:detect",
                           "--webcam-calib", str(Path(temp)/"missing.json"),
                           "--log", str(Path(temp)/"log.csv"), "--metrics", str(Path(temp)/"m3.json")])
            self.assertEqual(status, 0)
            result = json.loads((Path(temp)/"m3.json").read_text(encoding="utf-8"))
            self.assertEqual(result["scope"], "capture_to_decision")
            self.assertEqual(result["n_frames"], 3)
            self.assertEqual(released, [True])

    def test_render_seek_record_and_resource_cleanup_with_fake_fe(self):
        calls, writes, released = [], [], []
        keys = iter([2555904, ord("g"), ord(" "), ord("q")])
        writer = SimpleNamespace(isOpened=lambda: True, write=lambda image: writes.append(image),
                                 release=lambda: released.append(True))
        cv2 = SimpleNamespace(imshow=lambda *args: None, waitKeyEx=lambda delay: next(keys),
                              getWindowProperty=lambda *args: 1, WND_PROP_VISIBLE=0,
                              VideoWriter=lambda *args: writer, VideoWriter_fourcc=lambda *args: 0,
                              destroyAllWindows=lambda: released.append("windows"))
        canvas = np.zeros((100, 200, 3), np.uint8)
        def render(state, opts):
            calls.append((state["t"], opts["show_gt"], state["tracks"][0]["id"]))
            return canvas
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"cv2": cv2}), \
                patch("src.app._callable", return_value=render), redirect_stdout(io.StringIO()):
            status = main(["--source", "synthetic", "--record", str(Path(temp)/"record.mp4"),
                           "--log", str(Path(temp)/"log.csv"), "--metrics", str(Path(temp)/"m3.json")])
            self.assertEqual(status, 0)
        self.assertEqual([call[0] for call in calls], [0, .1, .1, .2])
        self.assertTrue(calls[2][1])
        self.assertEqual({call[2] for call in calls}, {1})
        self.assertTrue(all(frame is canvas for frame in writes))
        self.assertEqual(len(writes), 2)
        self.assertEqual(released, [True, "windows"])


class SourceTests(unittest.TestCase):
    def test_webcam_calibration_resizes_intrinsics(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"webcam.json"
            path.write_text(json.dumps(dict(K=[[100, 0, 50], [0, 100, 40], [0, 0, 1]],
                                            size=[100, 80], dist=[0]*5)), encoding="utf-8")
            calib, estimated = load_webcam_calib(path, (200, 160))
            self.assertEqual(calib.K[0, 0], 200)
            self.assertEqual(calib.K[1, 2], 80)
            self.assertFalse(estimated)
            np.testing.assert_allclose(calib.T_cam2ego @ [0, 0, 1, 0], [1, 0, 0, 0])

    def test_nuscenes_camera_motion_and_ground_truth_frame(self):
        identity = dict(rotation=[1, 0, 0, 0], translation=[0, 0, 0])
        records = {
            ("sample", "sample1"): dict(data={"LIDAR_TOP": "lidar", "CAM_FRONT": "camera"},
                                         anns=["ann"], next="", timestamp=1_000_000),
            ("sample_data", "lidar"): dict(ego_pose_token="ref"),
            ("sample_data", "camera"): dict(ego_pose_token="cam_pose", calibrated_sensor_token="calib", filename="image.jpg"),
            ("ego_pose", "ref"): dict(identity, translation=[10, 0, 0]),
            ("ego_pose", "cam_pose"): dict(identity, translation=[11, 0, 0]),
            ("calibrated_sensor", "calib"): dict(identity, camera_intrinsic=np.eye(3).tolist()),
            ("sample_annotation", "ann"): dict(identity, category_name="vehicle.motorcycle", translation=[15, 2, 1],
                                                size=[1, 2, 1.5], visibility_token="3", instance_token="rider1"),
        }
        nusc = SimpleNamespace(scene=[dict(name="test", token="scene", first_sample_token="sample1")],
                               get=lambda table, token: records[(table, token)])
        source = NuScenesSource("unused", "test", cameras=["CAM_FRONT"], nusc=nusc)
        cv2 = SimpleNamespace(imread=lambda path: np.zeros((80, 100, 3), np.uint8))
        with patch.dict("sys.modules", {"cv2": cv2}):
            frame = next(source)
            self.assertEqual(frame.calib["CAM_FRONT"].T_cam2ego[0, 3], 1)
            self.assertEqual(frame.gt[0]["x"], 5)
            self.assertEqual(frame.gt[0]["y"], 2)
            self.assertEqual(frame.gt[0]["l"], 2)
            self.assertEqual(frame.gt[0]["cls"], "rider")
            self.assertEqual(frame.t, 1)
            with self.assertRaises(StopIteration):
                next(source)
            source.seek(0)
            self.assertEqual(next(source).frame, 0)


if __name__ == "__main__":
    unittest.main()
