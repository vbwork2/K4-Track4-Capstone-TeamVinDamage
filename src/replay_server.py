"""Local replay server for the React FE: pick a scene, seek, play.

Runs the real pipeline (cache -> Core geometry -> BE fusion/tracker/zones) once per scene,
keeps every frame state, then serves single frames and camera JPEGs on demand.
Also accepts live states (POST /state, GET /state) like frontend/tools/vindamage_bridge.py.

  python -m src.replay_server [--port 8765] [--cache cache] [--phenikaa data/phenikaa] [--nuscenes data/mini-nuScenes]

GET /scenes                      -> [{id, dataset, scene, model, n_frames, period_s, ready}]
GET /frame?scene=<id>&i=<n>      -> FE state; images are URLs to /img (first call per scene runs the pipeline)
GET /img/<id>/<n>/<cam>.jpg      -> camera JPEG, width --img-width
Loopback only: there is no authentication.
"""

import argparse
import json
import threading
import time
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .cache import DetectionCache
from .config import load_config
from .pipeline import Pipeline

PERIOD = {"phenikaa": 0.1, "nuscenes": 0.5}
URL_PREFIX = "/warning-api"  # Vite proxies /warning-api/* to this server and strips the prefix


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Scene:
    def __init__(self, sid, dataset, scene, model, root):
        self.id, self.dataset, self.scene, self.model, self.root = sid, dataset, scene, model, root
        self.states, self.paths, self.source = None, None, None
        self.lock = threading.Lock()
        self.lidar_memo = OrderedDict()

    def info(self):
        n = len(self.states) if self.states is not None else None
        return dict(id=self.id, dataset=self.dataset, scene=self.scene, model=self.model,
                    n_frames=n, period_s=PERIOD[self.dataset], ready=self.states is not None)

    def _make_source(self):
        from . import sources
        if self.dataset == "phenikaa":
            return sources.PhenikaaSource(self.root, self.scene)
        return sources.NuScenesSource(self.root, self.scene)

    def ensure(self):
        """Run the full pipeline once; image decoding is skipped (only image sizes are needed)."""
        with self.lock:
            if self.states is not None:
                return
            import cv2
            from PIL import Image
            from . import geometry
            t0 = time.perf_counter()
            reads, sizes = [], {}
            real_imread = cv2.imread
            zero = np.zeros((1, 1, 3), np.uint8)

            def fake_imread(path, *args):
                path = str(path)
                reads.append(path)
                if path not in sizes:
                    sizes[path] = Image.open(path).size
                w, h = sizes[path]
                return np.broadcast_to(zero, (h, w, 3))

            src = self._make_source()
            cache = DetectionCache(Path(CACHE) / self.dataset / self.scene / f"{self.model}.jsonl")
            cfg = load_config(self.dataset)
            pipe = Pipeline(cfg, geometry.ground_point, geometry.height_point, self.dataset, self.scene, self.model)
            states, paths = [], []
            cv2.imread = fake_imread
            try:
                for frame in src:
                    state = pipe.process(frame, cache.read(frame))
                    cams = list(frame.images)
                    paths.append(dict(zip(cams, reads[-len(cams):])))
                    states.append(_light_state(state, frame))
            finally:
                cv2.imread = real_imread
            self.states, self.paths, self.source = states, paths, src
            print(f"[replay] {self.id}: {len(states)} frames in {time.perf_counter()-t0:.1f}s", flush=True)

    def frame(self, i):
        self.ensure()
        i = max(0, min(len(self.states) - 1, i))
        out = dict(self.states[i])
        q = urllib.parse.quote(self.id, safe="")
        out["images"] = {cam: f"{URL_PREFIX}/img/{q}/{i}/{cam}.jpg" for cam in self.paths[i]}
        out["frame"], out["n_frames"] = i, len(self.states)
        points = self.lidar(i)
        if points is not None:
            out["lidar_points"] = points
        return out

    def lidar(self, i):
        if i in self.lidar_memo:
            return self.lidar_memo[i]
        try:
            pts = _phenikaa_lidar(self, i) if self.dataset == "phenikaa" else _nuscenes_lidar(self, i)
        except Exception as exc:
            print(f"[replay] LiDAR off for {self.id}: {exc}", flush=True)
            pts = None
        if pts is not None:
            keep = (np.abs(pts[:, 0]) < 40) & (np.abs(pts[:, 1]) < 40) & (pts[:, 2] > 0.2) & (pts[:, 2] < 3.0)
            pts = pts[keep]
            if len(pts) > 12000:
                pts = pts[np.random.default_rng(0).choice(len(pts), 12000, replace=False)]
            pts = np.round(pts, 2).tolist()
        self.lidar_memo[i] = pts
        while len(self.lidar_memo) > 64:
            self.lidar_memo.popitem(last=False)
        return pts


def _light_state(state, frame):
    rng = state["zone_cfg"].get("max_range_m", 30.0)
    dets = [{k: d[k] for k in ("cam", "cls", "conf", "xyxy", "track_id", "level", "uncertain") if k in d}
            for d in state["dets"]]
    # BEV shows tracks with a detection this frame inside the trusted range; coasting
    # (predicted_only) tracks stay in the BE log and in stats.n_tracks but would clutter the view.
    tracks = [{k: v for k, v in t.items() if k != "dets"} for t in state["tracks"]
              if (t["x"]**2 + t["y"]**2) ** 0.5 <= rng and "predicted_only" not in t.get("reasons", [])]
    sizes = {cam: [img.shape[1], img.shape[0]] for cam, img in frame.images.items()}
    out = {k: state[k] for k in ("t", "source", "scene", "model", "gt", "ego_box", "zone_cfg", "stats")}
    out.update(dets=dets, tracks=tracks, image_sizes=sizes)
    return json.loads(json.dumps(out, default=_default))


def _phenikaa_lidar(scene, i):
    import laspy
    t = scene.states[i]["t"]
    folder = Path(scene.root) / scene.scene / "lidar"
    files = sorted(folder.glob("*.laz"))
    stamps = np.array([float(f.stem.replace("-", ".", 1)) for f in files])
    j = int(np.argmin(np.abs(stamps - t)))
    if abs(stamps[j] - t) > 0.06:
        return None
    las = laspy.read(files[j])
    return np.stack([las.x, las.y, las.z], 1)


def _nuscenes_lidar(scene, i):
    from .sources import pose_matrix
    src = scene.source
    data = src.nusc.get("sample_data", src.samples[i]["data"]["LIDAR_TOP"])
    raw = np.fromfile(Path(src.root) / data["filename"], dtype=np.float32).reshape(-1, 5)[:, :3]
    T = pose_matrix(src.nusc.get("calibrated_sensor", data["calibrated_sensor_token"]))
    return raw @ T[:3, :3].T + T[:3, 3]


SCENES = OrderedDict()
CACHE = "cache"
IMG_W = 640
LIVE = {"state": None, "t": 0.0}
LIVE_LOCK = threading.Lock()
IMG_CACHE = OrderedDict()
IMG_LOCK = threading.Lock()


def discover(cache, roots, models=None):
    for ds_dir in sorted(Path(cache).glob("*")):
        ds = ds_dir.name
        if ds not in roots or not Path(roots[ds]).exists():
            continue
        for scene_dir in sorted(ds_dir.glob("*")):
            for f in sorted(scene_dir.glob("*.jsonl")):
                if models and f.stem not in models:
                    continue
                sid = f"{ds}|{scene_dir.name}|{f.stem}"
                SCENES[sid] = Scene(sid, ds, scene_dir.name, f.stem, roots[ds])


def _jpeg(scene, i, cam):
    import cv2
    key = (scene.id, i, cam)
    with IMG_LOCK:
        if key in IMG_CACHE:
            IMG_CACHE.move_to_end(key)
            return IMG_CACHE[key]
    path = scene.paths[i][cam]
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (IMG_W, round(h * IMG_W / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    data = buf.tobytes()
    with IMG_LOCK:
        IMG_CACHE[key] = data
        while len(IMG_CACHE) > 600:
            IMG_CACHE.popitem(last=False)
    return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", cache="no-store"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, default=_default).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        try:
            if url.path == "/scenes":
                order = sorted(SCENES.values(), key=lambda s: (s.dataset != "phenikaa", s.scene))
                return self._send(200, [s.info() for s in order])
            if url.path == "/frame":
                scene = SCENES.get(q.get("scene", [""])[0])
                if scene is None:
                    return self._send(404, {"error": "Unknown scene"})
                return self._send(200, scene.frame(int(q.get("i", ["0"])[0])))
            if url.path.startswith("/img/"):
                _, _, sid, i, cam = url.path.split("/", 4)
                scene = SCENES.get(urllib.parse.unquote(sid))
                if scene is None or scene.paths is None:
                    return self._send(404, {"error": "Scene not loaded"})
                return self._send(200, _jpeg(scene, int(i), cam.removesuffix(".jpg")), "image/jpeg", "max-age=3600")
            if url.path == "/state":
                with LIVE_LOCK:
                    state, age = LIVE["state"], time.monotonic() - LIVE["t"]
                if not state or age > 2.5:
                    return self._send(503, {"error": "Waiting for fresh backend state (<2.5s)"})
                return self._send(200, state)
            return self._send(404, {"error": "Not found"})
        except Exception as exc:
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        if self.path != "/state":
            return self._send(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 12 * 1024 * 1024:
                return self._send(413, {"error": "Body must be 1 to 12MB"})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or not isinstance(body.get("tracks"), list):
                return self._send(422, {"error": "Expected BE state dict with tracks"})
            with LIVE_LOCK:
                LIVE["state"], LIVE["t"] = body, time.monotonic()
            return self._send(200, {"ok": True})
        except ValueError as exc:
            return self._send(400, {"error": str(exc)})


def main():
    global CACHE, IMG_W
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--phenikaa", default="data/phenikaa")
    ap.add_argument("--nuscenes", default="data/mini-nuScenes")
    ap.add_argument("--models", help="comma list of cache model stems to expose (default: all)")
    ap.add_argument("--img-width", type=int, default=640)
    ap.add_argument("--preload", action="store_true", help="run the pipeline for every scene at start")
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        ap.error("Loopback only: the server has no authentication.")
    CACHE, IMG_W = a.cache, a.img_width
    discover(a.cache, {"phenikaa": a.phenikaa, "nuscenes": a.nuscenes},
             set(a.models.split(",")) if a.models else None)
    print(f"[replay] {len(SCENES)} scenes: {', '.join(SCENES)}", flush=True)
    order = sorted(SCENES.values(), key=lambda s: (s.dataset != "phenikaa", s.scene))
    if a.preload:
        for s in order:
            s.ensure()
    else:  # warm every scene in the background so the FE scene picker responds quickly
        def warm():
            for s in order:
                try:
                    s.ensure()
                except Exception as exc:
                    print(f"[replay] {s.id} failed: {type(exc).__name__}: {exc}", flush=True)
        threading.Thread(target=warm, daemon=True).start()
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[replay] http://{a.host}:{a.port}  (FE: npm run dev in frontend/, open /#replay)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
