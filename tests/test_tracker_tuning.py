import unittest

import numpy as np

from src.contracts import Observation
from src.tracker import Tracker

FIT = {"radial": [0.1, 0.009], "tangential": [0.05, 0.01]}


class TrackerNoiseModelTest(unittest.TestCase):
    def test_radial_noise_dominates_far(self):
        tracker = Tracker(noise_model=FIT)
        R = tracker._measurement_R(Observation("car", .9, 20.0, 0.0))
        self.assertAlmostEqual(np.sqrt(R[0, 0]), 0.1 + 0.009*400, places=6)  # along the ray (x)
        self.assertAlmostEqual(np.sqrt(R[1, 1]), 0.05 + 0.01*20, places=6)

    def test_far_radial_jump_keeps_track_with_mahalanobis_gate(self):
        for gate_chi2, expected_ids in ((None, 2), (16.0, 1)):
            tracker = Tracker(noise_model=FIT, gate_chi2=gate_chi2, gate_m=6.0 if gate_chi2 else 1.5)
            tracker.update([Observation("car", .9, 25.0, 1.0)], 0.0, np.eye(4))
            out = tracker.update([Observation("car", .9, 27.5, 1.1)], 0.1, np.eye(4))
            self.assertEqual(len({t["id"] for t in out if "predicted_only" not in t["reasons"]} |
                                 {t["id"] for t in out}), expected_ids)

    def test_lateral_jump_near_is_rejected(self):
        tracker = Tracker(noise_model=FIT, gate_chi2=16.0, gate_m=6.0)
        tracker.update([Observation("rider", .9, 3.0, 2.0)], 0.0, np.eye(4))
        out = tracker.update([Observation("rider", .9, 3.0, 4.5)], 0.1, np.eye(4))
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
