# Kế hoạch chi tiết vai Core (Khánh)

Đi kèm [SPEC.md](../SPEC.md) bản 2 và [PHAN_CONG.md](PHAN_CONG.md). File này chia việc Core thành từng bước có đầu ra và cách kiểm tra. Giao kèo với BE và FE giữ đúng PHAN_CONG §2.

**Core sở hữu:**
- `src/calib.py`, `src/geometry.py`, `src/detect.py`, `src/detect_cache.py`;
- `src/sources.py` (phần `PhenikaaSource`);
- `src/eval_bev.py`, `src/eval_zones.py`;
- `colab/`, `tools/` (dữ liệu, fine-tune), video V1–V3.

---

## 0. Đã biết trước (kiểm ngày 16/10, không phải đoán)

| Điều | Kết quả | Hệ quả |
|---|---|---|
| Chiều `Sensor_Extrinsics.json` | **LiDAR → camera.** Chiếu nhãn với ma trận giữ nguyên thì box khớp xe ở `CAM_P_F`, `CAM_P_L`, `CAM_P_B`. Đảo ngược thì box bay lên trời | `T_cam2ego = inv(E[cam])`. Lúc thi chỉ chạy lại 5 phút để chụp ảnh cho slide |
| Tâm `CAM_P_F` trong hệ LiDAR | (1,31; 0,00; 1,88) m | camera cao khoảng 1,9 m so với gốc LiDAR, khớp SPEC §4.1 |
| Mặt đường Phenikaa | đáy box nhãn trong 15 m có median z = −0,04 m | z_g ≈ 0 trong hệ LiDAR. Vẫn fit từ LiDAR để có số hiệu chuẩn; nếu fit lệch quá 0,15 m so với −0,04 thì nghi bước fit sai |
| Ảnh thừa | `CAM_P_B/1758620952-599742889(1).jpg` và `CAM_P_R/1758620954-799739838(1).jpg` (COMPLEX CITY) là bản sao y hệt | bỏ mọi file có `(` trong tên; đồng bộ theo timestamp |
| Mui xe | `CAM_P_F` thấy nắp capo đỏ ở khoảng 15% dưới ảnh | box có đáy nằm trong vùng capo là phát hiện nhầm; có thể bỏ qua khi lọc |
| `colab/_env.sh` | `H=/mnt/c/.../final_hackathon` (repo cũ) | **phải sửa** thành `/mnt/c/Users/Khanh/Documents/AInAction/K4-Track4-Capstone-TeamVinDamage` |
| nuScenes trong `data/` | chỉ 3 scene: 0553, 0796, 1094 (121 keyframe) | SPEC §6 M4 ghi "404 keyframe" và §7 L3 ghi 1077/1100: sửa spec thành 3 scene / 121 keyframe, L3 chỉ dùng 1094 (1077, 1100 nằm trong tập train fine-tune) |

Ảnh kiểm ngoại tham số nằm ở scratchpad. Lúc thi tạo lại bằng `tools/check_extrinsics.py` (mục 2.2).

---

## 1. Trước buổi (chuẩn bị môi trường, không viết code sản phẩm)

| # | Việc | Kiểm xong khi |
|---|---|---|
| P1 | Xin chủ dữ liệu Phenikaa đồng ý: đưa lên Colab, chiếu ảnh lên slide/video. Lưu tin nhắn đồng ý | có tin nhắn đồng ý |
| P2 | Sửa `colab/_env.sh`: đặt `H` về repo này | `wsl ... ls $H/src` chạy được |
| P3 | `up.sh`: thêm `WITH_PHENIKAA=1` (tar không nén 6 `CAM_P_*` + `label/` + `ego_pose/` của 2 scene + `Calib Information/`, `--exclude='*(*'`). Thêm `WITH_FT=1` (tar `data/finetune_nusc`). Thêm `tools/` vào gói code | chạy thử 1 lần, `ls /content/phenikaa` đủ 2 scene, rồi `stop.sh` |
| P4 | `bash colab/usage.sh`: còn đủ CU cho khoảng 2 giờ GPU | ghi số CU còn lại |
| P5 | venv local: `ultralytics opencv-python numpy filterpy nuscenes-devkit laspy[lazrs]`; tải `yolo11n.pt`, `yolo11m.pt` | `python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"` |
| P6 | Đưa `data/` (trừ `phenikaa/` và `demo_videos/phenikaa_*`) lên Drive chia sẻ | BE/FE tải được nuScenes |
| P7 | Đọc lại `tools/nusc_to_yolo.py` (hàm `box2d`). M2 dùng lại cách chiếu box 3D → 2D này | |

---

## 2. Trong 2 giờ: từng khối

Giờ theo lịch ban tổ chức (build 0:15–2:15). Cột "Xong khi" là điều kiện để chuyển sang việc tiếp theo.

### 2.1 0:00–0:15: khởi động

1. Chạy ngay (nền), vì upload khoảng 355 MB + 422 MB mất thời gian:
   ```
   $env:WSLENV="WITH_PHENIKAA/u:WITH_FT/u"; $env:WITH_PHENIKAA="1"; $env:WITH_FT="1"
   wsl -d Ubuntu -e bash /mnt/c/Users/Khanh/Documents/AInAction/K4-Track4-Capstone-TeamVinDamage/colab/up.sh
   ```
2. Khung repo: `src/__init__.py`, các file rỗng theo SPEC §13, `config/`, `results/`, `cache/`.
3. Chốt với BE (5 phút):
   - chữ ký `CamCalib`, `Hit`, `ground_point`, `height_point` (PHAN_CONG §2.2);
   - **cột log CSV** để `eval_zones` đọc: `frame, t, n_tracks, sat_suon_alert (0/1), sat_suon_uncertain (0/1)`;
   - tên hàm gộp rider trong `fuse.py` để `eval_bev` gọi lại: `merge_riders(dets) -> dets`.

**Xong khi:** `up.sh` đang chạy; repo có khung; BE đã đồng ý cột CSV.

### 2.2 0:15–0:35: calib + kiểm ngoại tham số + cache yolo11n

**`src/calib.py`**
```python
@dataclass
class CamCalib:
    name: str; K: np.ndarray; dist: np.ndarray; size: tuple[int, int]
    T_cam2ego: np.ndarray; z_ground: float

def load_calib(dataset, root, scene=None, sample_token=None) -> dict[str, CamCalib]
```
- Phenikaa:
  - `K`, `dist` lấy từ `Camera_Intrinsics.json` (5 hệ số cho F/FL/FR/B, 8 hệ số cho L/R; `cv2` nhận cả hai);
  - `size = (1920, 1536)`;
  - `T_cam2ego = inv(np.array(E[cam]))`;
  - `z_ground` đọc từ `config/phenikaa.json`, ghi ở bước 2.3.
- nuScenes: `calibrated_sensor` của camera (quaternion → R; hàm `quat_to_matrix` lấy từ P-115); `dist = zeros(5)`; `z_ground = 0`; `size = (1600, 900)`.
- Webcam: `calib/webcam.json` do FE làm; `T_cam2ego` = đơn vị (Live dùng f·H/h).

**`tools/check_extrinsics.py`** (Core, 10 phút)
- Chép logic từ scratchpad: khung 100 COMPLEX CITY, chiếu 8 góc box nhãn bằng `cv2.projectPoints` với K + dist.
- Vẽ `CAM_P_F` (giữ nguyên E) cạnh `CAM_P_F` (E đảo ngược), thêm `CAM_P_L` và `CAM_P_B`.
- Mở `COMPLEX_CITY_projected_3d_box_5x2.mp4` cùng lúc để so.
- Lưu `results/extrinsics_check.jpg` cho slide (chỉ dùng nếu chủ dữ liệu cho chiếu).

**Colab: cache yolo11n Phenikaa** (khi `SETUP_DONE` xuất hiện trong log)
```
$env:WSLENV="JOB/u"; $env:JOB="python -m src.detect_cache --dataset phenikaa --root /content/phenikaa --model yolo11n.pt --out cache/"
wsl -d Ubuntu -e bash .../colab/run.sh
```

**`src/detect.py`**
- `Detector(model_path, conf=0.25, imgsz=1280)`;
- `detect(images: dict[cam, ndarray]) -> dict[cam, list[{cls, conf, xyxy}]]`;
- một lần gọi `model.predict(list_6_ảnh, classes=[0,1,2,3,5,7], half=True)`;
- `cls` là tên COCO.
- Cache lưu conf ≥ 0,25, ngưỡng 0,40 áp lúc đọc. Như vậy đổi ngưỡng hay vẽ đường cong M2 không cần chạy lại Colab.

**`src/detect_cache.py`**
- Duyệt khung theo `PhenikaaSource` / `NuScenesSource`. Viết bản tối thiểu chỉ đọc ảnh, không cần calib, để không phải chờ BE.
- Ghi đúng PHAN_CONG §2.1: một dòng mỗi (khung, camera) gồm `t`, `frame`, `cam`, `boxes`.
- Ghi thêm `results/m3_colab_<dataset>_<model>.json`:
  - bỏ 10 khung đầu (khởi động);
  - đo ms của mỗi lần gọi 6 camera;
  - báo `p50`, `p95`, tên GPU, `imgsz`.

**Xong khi:**
- ảnh kiểm ngoại tham số khớp;
- job cache đang chạy;
- `load_calib("phenikaa")` trả về 6 camera, mỗi `T_cam2ego[:3, 3]` gần đúng vị trí trên xe (F ở x ≈ +1,3; B ở x < 0; L ở y > 0; R ở y < 0).

### 2.3 0:35–0:55: geometry + z_g + PhenikaaSource

**`src/geometry.py`**
```python
@dataclass
class Hit: x: float; y: float; dist: float; method: str; at_edge: bool

def pixel_ray(u, v, cam) -> np.ndarray          # undistortPoints(P=None) → [xn, yn, 1] → R_c2e @ ray (hệ xe)
def ground_point(xyxy, cam) -> Hit | None
def height_point(xyxy, cam, H) -> Hit | None
def project_ego(P_ego: (N,3), cam) -> (N,2), mask_in_front   # dùng cho M2, check, lớp đáp án
```
- `ground_point`:
  1. Lấy điểm giữa cạnh đáy box: u = (x0 + x1) / 2, v = y1.
  2. Khử méo, ra tia `d` trong hệ xe; gốc tia `o = T_cam2ego[:3, 3]`.
  3. Cắt mặt đường: s = (z_g − o_z) / d_z.
  4. Trả `None` nếu d_z ≥ −1e-6 (tia không chúi xuống đất) hoặc s ≤ 0.
  5. Tính x, y = o + s·d; dist = hypot(x, y).
  6. `at_edge = y1 ≥ H_img − 3` (rộng ra thì thêm `x0 ≤ 3` hoặc `x1 ≥ W_img − 3`).
- `height_point`:
  1. Lấy tia qua tâm box.
  2. Độ sâu theo trục camera là Z = f_y · H / (y1 − y0).
  3. Điểm trong hệ camera là Z · [xn, yn, 1], đổi sang hệ xe.
  4. Trả `method="height"`.
- Quy tắc chọn (cho BE gọi): nếu `at_edge` đáy thì dùng `height_point` với H theo lớp (SPEC §4.2), ngược lại dùng `ground_point`. Mọi điểm có `dist > 30` đều gắn cờ (BE dùng cho KHÔNG CHẮC).
- **Tự kiểm không cần YOLO** (5 phút, bắt buộc trước khi giao cho BE):
  1. Lấy tâm đáy box nhãn (x, y, z_g) ở 0–30 m.
  2. Dùng `project_ego` chiếu vào camera, ra (u, v).
  3. Gọi `ground_point` với box giả `[u−5, v−20, u+5, v]`.
  4. Sai số kỳ vọng < 0,05 m trên mọi camera Phenikaa và nuScenes. Lớn hơn thì sai ở khử méo hoặc ở chiều T.

**Fit z_g** (`tools/fit_ground.py`)
- `laspy` đọc 5 file `.laz` của COMPLEX CITY (khung 0, 50, 100, 150, 199).
- Giữ điểm có 3 m < r < 15 m và |y| < 8 m, bỏ hộp thân xe (`keep_mask`).
- Lấy 10% điểm thấp nhất mỗi ô 1 m.
- Fit mặt phẳng z = a·x + b·y + c bằng RANSAC, ngưỡng 5 cm.
- Ghi `config/phenikaa.json: {"z_ground": c, "tilt_deg": ...}`.
- So với số tham chiếu −0,04 m. Nếu độ nghiêng > 2° thì ghi chú, vẫn dùng mặt phẳng ngang ở z = c.
- Làm cho cả URBAN; nếu hai scene lệch nhau < 5 cm thì dùng chung.

**`PhenikaaSource`** (trong `src/sources.py`)
- Trục thời gian là 200 timestamp của `label/`.
- Mỗi camera chọn ảnh gần nhất theo timestamp tên file, bỏ file có `(`, lệch tối đa 50 ms. Lệch hơn thì để camera đó trống và ghi cảnh báo.
- `ego_T`: đọc 12 số → 3×4 → thêm hàng [0, 0, 0, 1]. Kiểm hướng bằng cách xem quãng đường giữa khung 0 và khung 199 có hợp lý không (khoảng 0–200 m trong 20 s).
- `gt`: đọc nhãn `x y z l w h yaw cls`. Đổi tên lớp (`Rider` → `rider`, `Pedestrian` → `person`, `Car` → `car`). Hệ xe = hệ LiDAR, không cần đổi tọa độ.
- `__iter__` trả `Frame(t, images, calib, ego_T, gt)`; tham số `load_images=False` để eval chạy nhanh.

**Tải cache về** bằng `colab/dl.sh`. Mở thử: số dòng = 200 × 6 × 2 scene = 2.400.

**Xong khi (mốc 0:55):**
- tự kiểm `geometry` đạt;
- vẽ nhanh chấm `ground_point` của detection `CAM_P_F` khung 100 lên một BEV matplotlib cạnh chấm nhãn: chấm gần khớp trong 10 m.

### 2.4 0:55–1:15: ghép trên máy demo

- Thay stub `geometry` của BE bằng bản thật. Chạy `python -m src.app --source phenikaa --root data/phenikaa --scene "COMPLEX CITY" --cache cache/`.
- Lỗi hay gặp:
  - tên camera không khớp giữa cache và source → in tập tên ra so;
  - `t` cache khác `t` source → ghép theo `frame`, không theo `t`.
- Live thô: BE lo. Core chỉ đưa `height_point` + `calib/webcam.json`.

**Xong khi (1:15):** Replay Phenikaa hiện chấm của 6 camera trên BEV; Live hiện chấm phía trước.

### 2.5 1:15–1:40: M2 + cache phần còn lại + V1

**Colab (chạy nền, xếp hàng trong một JOB):**
```
python -m src.detect_cache --dataset phenikaa --root /content/phenikaa --model yolo11m.pt --out cache/ &&
python -m src.detect_cache --dataset nuscenes --root /content/nusc --model yolo11n.pt --out cache/ &&
python -m src.detect_cache --dataset nuscenes --root /content/nusc --model yolo11m.pt --out cache/
```
- Gói nuScenes đưa lên Colab phải có 6 `CAM_*` của 3 scene, không chỉ `CAM_FRONT`: sửa `up.sh` (bước P3).
- Mỗi lệnh cũng ghi M3 Colab.

**`src/eval_bev.py` (M2):** chạy local, đọc cache.
1. Với mỗi khung và mỗi camera, chiếu box nhãn bằng `project_ego` (8 góc, có méo) → khung bao 2D, rồi lọc:
   - nuScenes: `visibility ≥ 2`;
   - Phenikaa: ≥ 50% diện tích box chiếu nằm trong ảnh, cao ≥ 8 px, mọi góc nằm trước camera;
   - bỏ nhãn có tâm cách xe > 30 m.
2. Detection: lọc conf ≥ 0,40, rồi `merge_riders`. Box rider lấy **hợp** của box person và box xe. Lý do: box `Rider` 3D bao cả người, nếu chỉ dùng box xe máy thì IoU < 0,5 rất nhiều.
3. Ghép trên ảnh:
   - nhóm lớp: rider ↔ Rider / motorcycle / bicycle; person ↔ Pedestrian / pedestrian; car ↔ Car / car / bus / truck;
   - ma trận IoU, ghép tham lam theo IoU giảm dần, IoU ≥ 0,5.
4. Mỗi cặp ghép được:
   - `est = geometry.choose(det)` (ground hoặc height);
   - `gt` = tâm đáy box nhãn (x, y);
   - `err = ||est − gt||`.
5. Gom theo:
   - dải khoảng cách của gt: 0–5, 5–10, 10–20, 20–30 m;
   - nhóm camera: Phenikaa trước = F/FL/FR, hông = L/R, sau = B; nuScenes trước = FRONT, hông = FRONT_LEFT/FRONT_RIGHT/BACK_LEFT/BACK_RIGHT, sau = BACK;
   - phương pháp (ground / height);
   - ngày/đêm (nuScenes: scene-1094 là đêm).
6. Ghi `results/m2_<dataset>_<model>.json`:
   - mỗi nhóm có `n`, `mean`, `median`, `p90`;
   - thêm `n_gt`, `n_matched` (recall ghép), `n_det_unmatched`.
7. In một bảng markdown để dán vào slide.

**Thứ tự chạy M2:** Phenikaa yolo11n trước, vì đây là số bắt buộc. Có cache thì chạy tiếp Phenikaa yolo11m, nuScenes yolo11n, nuScenes yolo11m.

**Kiểm M2 hợp lý:**
- sai số ở 0–5 m phải < 0,5 m;
- sai số tăng theo khoảng cách;
- nhóm hông (L/R, f ≈ 871, méo mạnh) sai lớn hơn nhóm trước.

Ngược lại thì xem lại khử méo.

**Video V1:** `tools/render_v1.py`
- Lưới 2×3 Phenikaa: box YOLO màu, box nhãn chiếu vào màu trắng.
- 200 khung COMPLEX CITY ở 10 fps.
- `encode_mp4` lấy từ P-115 (ghi nguồn).
- Chạy local, không cần Colab.

**Xong khi:** có `m2_phenikaa_yolo11n.json`; job Colab nuScenes đang chạy; có V1.

### 2.6 1:40–2:00: M4, L3, chọn ca lỗi, tắt Colab

**`src/eval_zones.py` (M4)**
1. Chạy `app.py --headless --log results/log_<dataset>_<model>.csv` trên cả scene (BE thêm cờ `--headless`; nếu chưa có thì gọi thẳng vòng lặp của app không vẽ).
2. Đáp án mỗi khung: footprint nhãn (hình chữ nhật xoay theo l, w, yaw) giao với vùng SÁT SƯỜN. Vùng lấy từ `config/` qua `zones.py` của BE, dùng `shapely` hoặc tự viết kiểm giao đa giác.
3. Hệ thống: lấy cột `sat_suon_alert`.
4. Tính TP, FP, FN, precision, recall theo khung, ra hai bản:
   - tính tất cả các khung;
   - bỏ khung có `sat_suon_uncertain = 1`.
5. Phenikaa có 400 khung (2 scene × 200). nuScenes có 121 keyframe (3 scene).
6. Ghi `results/m4_<dataset>_<model>.json`.

**L3:** lấy từ M2 nuScenes, tách 0553 + 0796 (ngày) và 1094 (đêm). Báo `n_matched / n_gt` (recall) và sai số theo từng nhóm. Recall đêm thấp là con số chính của L3.

**Chọn khung ca lỗi** (cùng BE, 10 phút): viết `tools/find_cases.py` lọc log CSV và M2.
- **L1:** khung có > 15 rider trong 15 m phía trước và nhiều nhãn không ghép được.
- **L2:** khung có detection `CAM_P_L/R` với `at_edge` = true và nhãn Rider cách thân xe < 1,5 m.
- **L4:** cặp detection từ F và FL/FR cùng lớp, lệch nhau 1–2,5 m trên BEV.

Ghi số khung vào `results/cases.json`; FE gắn phím nhảy tới các khung này.

**Tắt Colab:** `dl.sh` → kiểm `results/` đủ file → `stop.sh` → `usage.sh` xác nhận không còn session.

**Xong khi:** có M4 Phenikaa; có số L3; có `cases.json`; Colab đã tắt.

### 2.7 2:00–2:30: đóng băng, video

- **V2:** `python -m src.app --source phenikaa --root data/phenikaa --scene "COMPLEX CITY" --cache cache/ --record v2.mp4`. Quay 3 đoạn theo `cases.json` (SÁT SƯỜN, TẠT ĐẦU, L2).
- **V3:** `python -m src.app --source webcam --record v3.mp4`. Chỉ quay thành viên đội; đi về phía webcam, rồi ngồi xuống (L5).
- Mở thử cả 3 video trên máy demo.
- Dán bảng M2/M3/M4 cho FE. Ô nào chưa đo thì ghi "chưa đo".

---

## 3. Fine-tune nhẹ YOLO (phần thêm, có thể bỏ)

Chi tiết ở [FINETUNE.md](FINETUNE.md). Dữ liệu đã có sẵn ở `data/finetune_nusc/` (422 MB):
- train: 6 scene nuScenes không dùng để đo;
- val: scene-0061;
- test: đúng 3 scene M2/M4;
- id lớp COCO.

Khớp vào lịch như sau:
- **0:15:** bắt đầu chạy nền `python tools/ft_yolo.py --data /content/finetune_nusc/data.yaml --model yolo11n.pt --out results/`. Nếu GPU đang bận cache thì chạy **sau** cache yolo11n Phenikaa.
- **1:15:** đọc `results/ft_yolo11n.json`. Chỉ khi mAP50 test tăng mới chạy `detect_cache --model yolo11n_ft.pt` cho Phenikaa và thêm một cột M2.
- Không dùng LoRA đúng nghĩa (Ultralytics không hỗ trợ cho CNN), mà đóng băng backbone (`freeze=10`). Nói đúng như vậy khi bị hỏi.
- Nếu dùng thì sửa SPEC §8 dòng "Học".

---

## 4. Nếu chậm (Core)

| Trễ ở mốc | Bỏ | Giữ bằng mọi giá |
|---|---|---|
| 0:35 (`up.sh` chưa xong) | chạy cache yolo11n Phenikaa **local** trên RTX 4050 (khoảng 2.400 ảnh, vài phút) | cache Phenikaa yolo11n |
| 0:55 | `height_point` (chỉ ground), fit z_g (dùng −0,04 m từ nhãn, ghi rõ) | `ground_point`, `PhenikaaSource` |
| 1:40 | fine-tune, yolo11m, nuScenes 6 camera (chỉ `CAM_FRONT` cho L3), V1 | M2 Phenikaa yolo11n |
| 2:00 | M4 nuScenes, L4 | M4 Phenikaa, V2 |

---

## 5. Kiểm nhanh trước demo

- [ ] `results/`: `m2_phenikaa_yolo11n.json`, `m3_colab_*.json`, `m4_phenikaa_yolo11n.json`, `cases.json`, `extrinsics_check.jpg`
- [ ] Bảng M2 có số cặp ghép được (n) cạnh mỗi sai số
- [ ] `stop.sh` đã chạy; `usage.sh` không còn session
- [ ] `git status` không có `data/`, `cache/`, `.env`, `*.pt`, `*.mp4`
- [ ] Ảnh Phenikaa chỉ lên slide/video khi đã có đồng ý (P1)
- [ ] V1, V2, V3 phát được trên máy demo

---

## 6. Trả lời sẵn câu hỏi cho Core

- **Vì sao điểm tiếp đất mà không f·H/h?**
  - Điểm tiếp đất cho cả x lẫn y và không cần biết chiều cao vật.
  - Rider cao từ 1,4 đến 1,8 m, sai 10% chiều cao là sai 10% khoảng cách.
  - f·H/h chỉ dùng khi đáy box bị cắt.
- **Kiểm ngoại tham số thế nào?**
  - Chiếu box nhãn LiDAR vào ảnh bằng ma trận giữ nguyên và ma trận đảo ngược.
  - Bản giữ nguyên khớp vật và khớp video chiếu sẵn của chủ dữ liệu; bản đảo ngược bay ra ngoài.
  - Tâm camera trước ra (1,3; 0; 1,9) m, hợp lý với vị trí trên nóc xe.
- **Vì sao 30 m?**
  - Camera cao khoảng 1,9 m. Ở 30 m, tia chỉ chúi khoảng 3,6° so với mặt đường.
  - Lệch 1 px theo chiều dọc ở f = 1845 tương đương khoảng 0,25 m. Đường không phẳng ±5 cm ra thêm khoảng 0,8 m.
  - M2 dải 20–30 m cho thấy sai số tăng.
- **Vì sao M2 ghép trên ảnh?**
  - Nếu ghép trên BEV thì ghép theo khoảng cách gần nhất, mà khoảng cách lại chính là thứ đang đo.
  - Khi đó sai số bị đánh giá thấp, vì cặp sai xa không ghép được và bị loại khỏi thống kê.
- **yolo11n vs yolo11m?** Đọc từ M3 (ms mỗi khung 6 camera) và M2 (recall ghép, sai số theo dải). Không nói trước số.
