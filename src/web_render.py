"""Renderer that sends BE state to the React FE through frontend/tools/vindamage_bridge.py.

Use: python -m src.app --source phenikaa ... --renderer src.web_render:render
Env:
  VD_BRIDGE   bridge URL (default http://127.0.0.1:8765/state); empty disables posting
  VD_EXPORT   optional path: also write every frame to a JSON array for the FE "Nhập JSON" mode
  VD_LIDAR    Phenikaa data root for the LiDAR BEV layer (default data/phenikaa); empty disables
  VD_IMG_W    camera JPEG width sent to the browser (default 640)
BEV tracks are limited to zone_cfg.max_range_m. Returns a small 3x2 camera grid so --record still writes an MP4.
"""

import atexit
import base64
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

BRIDGE = os.environ.get("VD_BRIDGE", "http://127.0.0.1:8765/state")
EXPORT = os.environ.get("VD_EXPORT", "")
LIDAR_ROOT = os.environ.get("VD_LIDAR", "data/phenikaa")
IMG_W = int(os.environ.get("VD_IMG_W", "640"))
COLORS = {0: (200, 200, 200), 1: (0, 190, 255), 2: (0, 0, 255)}

_export = None
_lidar_index = {}
_warned = set()


def _warn(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(f"[web_render] {msg}")


def _jpeg(image):
    import cv2
    h, w = image.shape[:2]
    small = cv2.resize(image, (IMG_W, round(h * IMG_W / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


def _lidar(state):
    """Nearest Phenikaa .laz sweep, ground removed, subsampled; ego frame = LiDAR frame."""
    if state["source"] != "phenikaa" or not LIDAR_ROOT:
        return None
    scene = state["scene"]
    if scene not in _lidar_index:
        folder = Path(LIDAR_ROOT) / scene / "lidar"
        files = sorted(folder.glob("*.laz")) if folder.is_dir() else []
        stamps = [float(f.stem.replace("-", ".", 1)) for f in files]
        _lidar_index[scene] = (np.array(stamps), files, {})
    stamps, files, memo = _lidar_index[scene]
    if not files:
        return None
    i = int(np.argmin(np.abs(stamps - state["t"])))
    if abs(stamps[i] - state["t"]) > 0.06:
        return None
    if i not in memo:
        try:
            import laspy
            las = laspy.read(files[i])
            pts = np.stack([las.x, las.y, las.z], 1)
        except Exception as exc:  # laspy or LAZ backend missing
            _warn("lidar", f"LiDAR layer off: {exc}")
            memo[i] = None
            return None
        keep = (np.abs(pts[:, 0]) < 40) & (np.abs(pts[:, 1]) < 40) & (pts[:, 2] > 0.2) & (pts[:, 2] < 3.0)
        pts = pts[keep]
        if len(pts) > 12000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 12000, replace=False)]
        memo.clear()  # keep only the current sweep in memory
        memo[i] = np.round(pts, 2).tolist()
    return memo[i]


def _payload(state):
    images, sizes = {}, {}
    for cam, image in state["images"].items():
        url = _jpeg(image)
        if url:
            images[cam] = url
        sizes[cam] = [image.shape[1], image.shape[0]]
    dets = [{k: d[k] for k in ("cam", "cls", "conf", "xyxy", "track_id", "level", "uncertain") if k in d}
            for d in state["dets"]]
    out = {k: state[k] for k in ("t", "source", "scene", "model", "gt", "ego_box", "zone_cfg", "stats")}
    # BEV shows the trusted range only; farther tracks are uncertain by design (zone_cfg.max_range_m)
    rng = state["zone_cfg"].get("max_range_m", 30.0)
    out["tracks"] = [t for t in state["tracks"] if (t["x"]**2 + t["y"]**2) ** 0.5 <= rng]
    out.update(images=images, image_sizes=sizes, dets=dets)
    points = _lidar(state)
    if points:
        out["lidar_points"] = points
    return out


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_export(text):
    global _export
    if _export is None:
        Path(EXPORT).parent.mkdir(parents=True, exist_ok=True)
        _export = open(EXPORT, "w", encoding="utf-8")
        _export.write("[\n")
        atexit.register(lambda: (_export.write("\n]\n"), _export.close()))
    else:
        _export.write(",\n")
    _export.write(text)


def _grid(state):
    import cv2
    tiles = []
    for cam, image in state["images"].items():
        tile = image.copy()
        for d in state["dets"]:
            if d.get("cam") != cam:
                continue
            x0, y0, x1, y1 = map(int, d["xyxy"])
            color = (0, 200, 255) if d.get("uncertain") else COLORS.get(d.get("level", 0), COLORS[0])
            cv2.rectangle(tile, (x0, y0), (x1, y1), color, 4)
        tile = cv2.resize(tile, (480, round(image.shape[0] * 480 / image.shape[1])))
        cv2.putText(tile, cam, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) < 6:
        tiles.append(np.zeros_like(tiles[0]) if tiles else np.zeros((384, 480, 3), np.uint8))
    h = min(t.shape[0] for t in tiles)
    tiles = [t[:h] for t in tiles[:6]]
    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])


def render(state, opts):
    text = json.dumps(_payload(state), default=_default, ensure_ascii=False)
    if BRIDGE:
        try:
            req = urllib.request.Request(BRIDGE, text.encode("utf-8"), {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=1.0).read()
        except Exception as exc:
            _warn("bridge", f"bridge not reachable at {BRIDGE} ({exc}); start frontend/tools/vindamage_bridge.py")
    if EXPORT:
        _write_export(text)
    return _grid(state)
