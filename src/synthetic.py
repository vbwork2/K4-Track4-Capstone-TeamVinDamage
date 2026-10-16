"""Explicitly synthetic inputs for integration checks without dataset or GPU."""

import numpy as np

from .contracts import CamCalib, Frame, Hit


class SyntheticSource:
    scene = "synthetic_not_measured"
    estimated_calib = True

    def __init__(self, count=60):
        self.count = count
        self.index = 0
        self.names = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
        self.calib = {cam: CamCalib(np.eye(3), np.zeros(5), (320, 240), np.eye(4)) for cam in self.names}

    def __len__(self):
        return self.count

    def seek(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.index = index

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.count:
            raise StopIteration
        index = self.index
        self.index += 1
        images = {cam: np.zeros((240, 320, 3), dtype=np.uint8) for cam in self.names}
        return Frame(index/10, images, self.calib, np.eye(4), [], index)

    def close(self):
        pass


def synthetic_boxes(frame):
    # The fake image x coordinate encodes distance for the stub geometry only.
    distance = 12 - frame.frame*0.15
    box = [100+distance, 100., 140+distance, 200.]
    return {cam: ([dict(cls="car", conf=0.9, xyxy=box, cam=cam)]
                  if cam == "CAM_FRONT" else []) for cam in frame.images}


def stub_ground_point(xyxy, cam):
    x = float(xyxy[0]-100)
    return Hit(x, 0.0, abs(x), "ground", False)


def stub_height_point(xyxy, cam, H):
    hit = stub_ground_point(xyxy, cam)
    hit.method = "height"
    return hit
