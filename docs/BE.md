# Backend implementation

This document describes the BE modules required by SPEC version 2 and
PHAN_CONG. Data, calibration measurements, Core geometry, and FE rendering can
be supplied later. The backend does not require an HTTP server or database.

## Implemented modules

| Module | Responsibility |
|---|---|
| `src/contracts.py` | Shared `CamCalib`, `Hit`, `Frame`, observation, and render-state contracts |
| `src/config.py`, `config/` | Defaults, dataset settings, and explicit JSON overrides |
| `src/sources.py` | nuScenes keyframes, camera calibration, ego poses, GT conversion, webcam |
| `src/cache.py` | JSONL cache reader with frame/camera lookup and timestamp fallback |
| `src/fuse.py` | Rider pairing, cross-camera fusion, disagreement candidates |
| `src/tracker.py` | One-to-one nearest BEV association and constant-velocity Kalman tracking |
| `src/zones.py` | Side, cut-in, front/TTC, uncertainty suppression, per-zone counts |
| `src/pipeline.py` | Core geometry integration and FE state construction |
| `src/telemetry.py` | Frame/track CSV logs and local timing summaries |
| `src/app.py` | CLI, Replay/Live lifecycle, FE rendering hooks, recording and controls |
| `src/synthetic.py` | Clearly labeled synthetic inputs and geometry for integration checks |

Comments and docstrings in code are in English. No dataset, weights, or real
evaluation result is included.

## Run without data

Only NumPy is needed for the synthetic check and backend tests:

```powershell
python -m pip install numpy
python -m unittest discover -s tests -v
python -m src.app --source synthetic --headless --max-frames 60 --log results/log_synthetic.csv --metrics results/m3_local_synthetic.json
```

This exercises six input camera slots, synthetic boxes, fusion, tracking,
warning logic, and CSV output. The synthetic calibration is explicitly marked
estimated, so alarms cannot become red. The JSON summary contains
`synthetic: true`; its timings are not M1/M2/M3/M4 evidence for the real system.
The synthetic images are blank arrays, not a demo visualization.

## Install runtime dependencies

Use a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-be.txt
```

Configure a suitable PyTorch build separately if GPU inference is needed.
The backend does not download datasets or start model inference on import.
Ultralytics may obtain model weights on the first real Live run if the file
does not already exist. Place the weights locally before an offline demo.
Replay uses cached boxes and does not need local YOLO inference.

## Connect Core

Core adds `src/geometry.py` using these interfaces:

```python
from src.contracts import CamCalib, Hit

def ground_point(xyxy, cam: CamCalib) -> Hit | None:
    ...

def height_point(xyxy, cam: CamCalib, H: float) -> Hit | None:
    ...
```

Coordinates are meters, x forward, y left, z up. `Hit.dist` is the distance in
meters. Geometry owns point undistortion and camera-to-ego conversion.
The backend uses the cycle box ground point for a merged rider, and the full
rider box for height fallback. Replay uses height fallback only if the bottom
touches the image edge; a failed ground intersection remains uncertain instead
of inventing a point. Live always uses height geometry.

Core adds `PhenikaaSource(root, scene)` to its portion of `src/sources.py`.
It must be an iterator yielding `Frame` and expose `scene`. Optional `seek(i)`
and `__len__` enable single-frame navigation and model switching. Optional
`frame_period_s` sets the display period; the default Phenikaa period is 0.1 s.
Optional `close()` releases source resources. `frame.frame` must use the same
zero-based numbering as the detection cache.

Detection cache location:

```text
cache/<dataset>/<scene>/<model_stem>.jsonl
```

Each line is one camera/frame, including an empty `boxes` list when nothing
was detected:

```json
{"t": 1700000000.123, "frame": 0, "cam": "CAM_FRONT", "boxes": [{"cls": "motorcycle", "conf": 0.82, "xyxy": [20, 30, 100, 150]}]}
```

Pixel coordinates are from original images. Frame/camera keys take priority
and their timestamps are checked against the frame within 1 ms;
for rows without a matching frame key, timestamps must match within 1 ms.
Missing camera rows raise an error instead of silently implying no objects.
`--cache` can also point to a single JSONL file; model switching then requires
restarting with another file. The reader currently loads one scene/model cache
into RAM, not the entire dataset.

## Connect FE

FE implements `src/ui.py` with `render(state, opts) -> ndarray BGR`.
The returned image must have stable dimensions during recording.
`state` follows PHAN_CONG section 2.4. Additional fields are:

- Track: `zones`, `conf`, `reasons`, `velocity_valid`.
- Detection: `reasons`, and `ground_xyxy` for a merged rider.
- Stats: `estimated_calib`.

`opts` contains `show_gt`, `paused`, and `record`. The application routes keys
to backend actions; FE only needs to draw their state. Supported keys are
Space, left/right, g, m, r, q, and Escape. Pause and stepping are Replay-only.
Seeking replays all preceding frames without rendering to restore tracking
history, rather than applying a negative time step. This can be slow on long
scenes. Missing alternate model caches leave the current model selected.

The application writes exactly the BGR canvas returned by FE when recording
new frames. It does not record raw camera frames or duplicate paused frames.
Recording resumes to a new file after it has been stopped. `--record-fps` is
fixed-rate encoding (default 10); choose an appropriate value for Live.

## Commands once Core, FE, and data are available

```powershell
python -m src.app --source nuscenes --root data/mini-nuScenes --scene scene-1094 --cache cache
python -m src.app --source phenikaa --root data/phenikaa --scene "COMPLEX CITY" --cache cache --record results/replay.mp4
python -m src.app --source webcam --model yolo11n.pt --inference-device 0
```

Use `--headless` to skip FE rendering. Real sources still require Core geometry
even in headless mode. A different renderer can be selected with
`--renderer package.module:render`. A Core detector can be injected using
`--detector package.module:detect`, where `detect(image_bgr)` returns a list
of `{cls, conf, xyxy}` dictionaries. The default Live adapter uses Ultralytics.

## Calibration and measured parameters

FE supplies `calib/webcam.json`:

```json
{
  "K": [[900, 0, 640], [0, 900, 360], [0, 0, 1]],
  "dist": [0, 0, 0, 0, 0],
  "size": [1280, 720],
  "estimated": false
}
```

These numbers describe the file format only; they are not measured calibration.
`T_cam2ego` and `z_ground` are optional for Live. The default orientation maps
camera z forward to ego x, camera x right to ego minus y, and camera y down to
ego minus z. Intrinsics scale with the actual captured image dimensions.
When the file is absent, the source explicitly warns about estimated HFOV=70
degrees and flags detections uncertain. Live timestamps use `perf_counter()`
immediately after capture returns; driver buffering before that point is not
measured sensor latency.

Default vehicle dimensions and Kalman measurement variance are design
assumptions. Replace the Phenikaa ego box with measured P-115 dimensions and
offset. Do not report defaults as M1 or M2 results. The tracker accepts a
positive scalar positional variance in square meters or a positive definite
2 by 2 covariance. Load the measured value using `--config`:

```json
{
  "tracker": {"measurement_var": [[0.04, 0.0], [0.0, 0.09]]}
}
```

This covariance is an example of the format only. Use M1/M2 residuals for the
real value. Tracker state is kept in world coordinates to compensate ego
motion. Public `vx` and `vy` are relative to the current ego frame, including
yaw-rate compensation. A new track needs two observations before velocity
warnings are enabled. Unmatched predictions are flagged uncertain and expire
after `max_age_s`. Tracking assumes planar motion with negligible roll/pitch.

The nuScenes reader moves camera extrinsics into the reference LIDAR_TOP ego
pose to account for exposure time differences. GT centers and orientations
are transformed from world to that reference frame. Class groups are person,
rider, and car (including bus/truck). `visibility` and annotation tokens remain
available to Core evaluation. The source does not filter low-visibility labels;
M2 applies that filter in the evaluator.

## Warning decisions and unresolved design details

The numeric thresholds follow SPEC section 4.2. Uncertainty propagates from
low confidence, image clipping, range, camera disagreement, Kalman residual,
estimated webcam calibration, invalid geometry, and predicted-only tracks.
Uncertain tracks never receive level 2. Invalid geometry keeps its camera box
visible and uncertain but creates no false BEV point.

SPEC does not define red thresholds for side/cut-in or overlapping-zone
priority. Configurable side/cut-in levels default to 1. For overlapping zones,
the primary zone is chosen by severity, then cut-in, side, front. All active
zones are retained for counts. These are explicit implementation choices for
the team to confirm before evaluating results.

Cross-camera disagreement uses mutual nearest same-class candidates between
1.5 and 3 m. It is a geometric heuristic, not proof of identity: two nearby
different objects can be flagged. Such pairs stay separate. Fusion itself
uses a strict 1 m gate, never combines same-camera detections, and prevents
chained clusters whose endpoints exceed that gate.

## Logs and evaluation handoff

CSV contains one row per frame/track, or one empty-track row if there are no
tracks. It includes position, velocity, TTC, levels, uncertainty reasons, counts,
timings, and frame uncertainty. Use frame numbers to deduplicate rows before
computing a per-frame metric. `frame_uncertain` includes unprojectable camera
detections. The log is flushed each frame.

M3 Live latency covers capture-return to decision, including detection and BE
processing and excluding rendering. Final FPS covers capture reading through
rendering/recording, excluding user pause and Replay pacing. Replay timings
exclude Colab inference and are labeled `local_replay_processing`, not Live
or Colab M3. UI FPS is a recent processing estimate; the JSON is the aggregate
timing summary. If the operator changes model mid-session, timing samples are
grouped by their actual model in the summary.

For M1, coordinate with FE: capture 30 frames at each measured 1/2/3 m mark,
then compute errors and stationary variance from the corresponding track
positions. No M1 collection result is fabricated here. For M4, Core can import
`evaluate_tracks`, `in_side_zone`, and `alert_counts` from `src.zones`; GT
footprint/zone intersection and metric aggregation stay with `eval_zones.py`.
M2 and Colab M3 also remain Core responsibilities.

## Validation scope

The unit suite checks rider pairing, duplicate suppression, camera changes,
ego translation/rotation, Kalman outliers, track expiry, TTC boundaries,
uncertainty, cache failures, source transforms, and end-to-end synthetic logs.
Real dataset paths, hardware webcam behavior, model inference, FE controls,
and MP4 encoding require integration validation when those inputs are ready.
