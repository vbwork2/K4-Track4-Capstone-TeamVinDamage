"""YOLO detector wrapper (Ultralytics, lazy import). Keeps road-user classes only."""

KEEP = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


class Detector:
    def __init__(self, model_path="yolo11m.pt", device=None, conf=0.25, imgsz=None):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device, self.conf, self.imgsz = device, conf, imgsz
        self.keep = [i for i, n in self.model.names.items() if n in KEEP]

    def batch(self, images):
        if not images:
            return []
        kw = dict(conf=self.conf, classes=self.keep, verbose=False)
        if self.device is not None:
            kw["device"] = self.device
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        out = []
        for r in self.model.predict(list(images), **kw):
            names, rows = r.names, []
            if r.boxes is not None and len(r.boxes):
                for xyxy, c, k in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(),
                                      r.boxes.cls.cpu().tolist()):
                    rows.append({"cls": names[int(k)], "conf": float(c), "xyxy": xyxy})
            out.append(rows)
        return out

    def __call__(self, image_bgr):
        return self.batch([image_bgr])[0]


_default = None


def detect(image_bgr):
    """Hook for `--detector src.detect:detect`. Live uses yolo11n (RTX 4050); use Detector for 11m."""
    global _default
    if _default is None:
        _default = Detector("yolo11n.pt")
    return _default(image_bgr)
