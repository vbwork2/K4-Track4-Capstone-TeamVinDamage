import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUSC = ROOT / "data" / "mini-nuScenes"


class FakeDetector:
    def batch(self, images):
        return [[{"cls": "car", "conf": 0.91234, "xyxy": [1.04, 2.0, 30.0, 40.0]}] for _ in images]


class DetectCacheTest(unittest.TestCase):
    def test_roundtrip(self):
        try:
            import nuscenes  # noqa: F401
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("nuscenes-devkit/cv2 missing")
        if not NUSC.exists():
            self.skipTest("mini-nuScenes missing")
        from src.cache import DetectionCache
        from src.detect_cache import run, summarize
        from src.sources import NuScenesSource
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.jsonl"
            times = run(NuScenesSource(NUSC, "scene-1094"), FakeDetector(), path, max_frames=2)
            self.assertEqual(len(times), 2)
            self.assertEqual(summarize(times, warmup=0)["n_frames"], 2)
            cache = DetectionCache(path)
            src = NuScenesSource(NUSC, "scene-1094")
            for _ in range(2):
                frame = next(src)
                got = cache.read(frame)
                self.assertEqual(set(got), set(frame.images))
                box = got["CAM_FRONT"][0]
                self.assertEqual((box["cls"], box["conf"], box["xyxy"][0]), ("car", 0.912, 1.0))


if __name__ == "__main__":
    unittest.main()
