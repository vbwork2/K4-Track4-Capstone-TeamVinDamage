import math
import unittest

from src.eval_bev import bin_label, class_group, kalman_R, match_greedy, stats
from src.eval_zones import footprint_in_side_zone, score

EGO = {"length": 4.5, "width": 1.8, "x_offset": 0.0}
ZONE = {"side_margin_m": 1.0, "longitudinal_margin_m": 1.0}


class EvalTests(unittest.TestCase):
    def test_match_iou_and_class(self):
        dets = [dict(cls="person", xyxy=[0, 0, 10, 10]), dict(cls="car", xyxy=[50, 50, 60, 60])]
        gts = [dict(cls="person", xyxy=[1, 0, 11, 10]), dict(cls="person", xyxy=[50, 50, 60, 60])]
        self.assertEqual([(i, j) for i, j, _ in match_greedy(dets, gts)], [(0, 0)])

    def test_match_greedy_one_to_one(self):
        dets = [dict(cls="car", xyxy=[0, 0, 10, 10]), dict(cls="bus", xyxy=[0, 0, 10, 11])]
        gts = [dict(cls="car", xyxy=[0, 0, 10, 10])]
        self.assertEqual(len(match_greedy(dets, gts)), 1)
        self.assertEqual(class_group("truck"), "car")

    def test_bins_and_stats(self):
        self.assertEqual([bin_label(d) for d in (0, 4.9, 5, 25, 30, 99)],
                         ["0-5", "0-5", "5-10", "20-30", "30-inf", "30-inf"])
        s = stats([1, 2, 3])
        self.assertEqual((s["n"], s["mean"], s["median"]), (3, 2, 2))
        self.assertIsNone(kalman_R([(0, 0, 5)]))
        self.assertEqual(kalman_R([(1, 0, 5), (-1, 0, 5), (0, 1, 5), (0, 0, 50)])["n"], 3)

    def test_footprint_zone(self):
        # side zone: |y| in [0.9, 1.9], x in [-3.25, 3.25]
        inside = dict(x=0, y=1.4, l=1, w=1, yaw=0)
        far = dict(x=0, y=6, l=1, w=1, yaw=0)
        self.assertTrue(footprint_in_side_zone(inside, EGO, ZONE))
        self.assertFalse(footprint_in_side_zone(far, EGO, ZONE))
        # centre outside, long car whose edge reaches the zone
        long_car = dict(x=7, y=1.4, l=8, w=1.8, yaw=0)
        self.assertTrue(footprint_in_side_zone(long_car, EGO, ZONE))
        # rotated 90 deg: lies along y, spans y 1..5 -> touches zone
        rot = dict(x=0, y=3.5, l=5, w=1.8, yaw=math.pi/2)
        self.assertTrue(footprint_in_side_zone(rot, EGO, ZONE))

    def test_precision_recall(self):
        s = score([(True, True), (True, False), (False, True), (False, False), (True, True)])
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (2, 1, 1))
        self.assertAlmostEqual(s["precision"], 2/3)
        self.assertAlmostEqual(s["recall"], 2/3)
        self.assertAlmostEqual(s["f1"], 2/3)
        self.assertIsNone(score([(False, False)])["precision"])


if __name__ == "__main__":
    unittest.main()
