# VinFast Surround Rider Warning

Cảnh báo xe máy áp sát sườn, tạt đầu và nguy cơ phía trước bằng 6 camera quanh xe, đặt mọi vật lên bản đồ BEV 360° theo mét. Chế độ Replay đọc detection YOLO đã cache trên dữ liệu Phenikaa (Hà Nội) hoặc nuScenes mini. Chế độ Live dùng một webcam, chỉ có vùng phía trước. Tài liệu gốc là [SPEC.md](SPEC.md) bản 2.

## Video demo

[▶ Xem video demo: Replay Phenikaa COMPLEX URBAN trên web (31 s)](<demo/Recording 2026-10-16 173813.mp4>)

Video nằm ở `demo/Recording 2026-10-16 173813.mp4` và **không được commit lên GitHub**. Video có ảnh camera Phenikaa; SPEC §12 cho phép chiếu trong slide và video demo khi chủ dữ liệu đồng ý, nhưng không cho đưa vào repo (repo đang public). Muốn xem trên GitHub thì đưa video lên Drive riêng của nhóm và thay link ở trên.

Nội dung video: chọn scene trong menu, lưới 6 camera có box YOLO11m, BEV 360° có nền LiDAR, track và vùng cảnh báo, banner trạng thái (SÁT SƯỜN / TẠT ĐẦU / PHÍA TRƯỚC / KHÔNG CHẮC), tua bằng timeline.

## Chạy demo

Cần Python 3.10+ (numpy, opencv-python, nuscenes-devkit, laspy[lazrs] cho nền LiDAR Phenikaa), Node.js 20.19+, dữ liệu trong `data/` và cache detection trong `cache/` (xem [Dữ liệu](#dữ-liệu)).

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-be.txt laspy[lazrs] pillow
powershell -File tools/demo.ps1          # bật replay server + FE, mở http://localhost:5173/
```

Trên web: nút **Replay**, chọn **Scene** (Phenikaa COMPLEX CITY / URBAN, nuScenes 0553 / 0796 / 1094 đêm), `Space` phát hoặc dừng, `←` `→` từng khung, `G` bật đáp án nhãn LiDAR. Lần đầu server chạy pipeline cho mọi scene mất khoảng 35 s; menu ghi "đang xử lý…" trong lúc đó.

Chạy tay từng phần:

```powershell
python -m src.replay_server                       # http://127.0.0.1:8765 (chỉ loopback)
cd frontend; npm install; npm run dev             # http://localhost:5173/
python -m src.app --source phenikaa --root data/phenikaa --scene "COMPLEX CITY" --cache cache --model yolo11m --headless --log results/log_city.csv
python -m src.app --source webcam --model yolo11n.pt   # Live, cửa sổ OpenCV
```

Ghi MP4 dự phòng không cần web: thêm `--renderer src.web_render:render --record results/replay_city.mp4` vào lệnh `src.app` ở trên.

## Hệ thống

```
6 camera ──► YOLO11m (Colab, cache JSONL) ──► gộp người + xe thành rider
         ──► điểm tiếp đất: khử méo → tia → cắt mặt đường → hệ xe (mét)
         ──► gộp vật trùng giữa camera ──► Kalman [x, y, vx, vy], bù chuyển động xe
         ──► vùng SÁT SƯỜN / TẠT ĐẦU / PHÍA TRƯỚC (TTC) + cờ KHÔNG CHẮC
         ──► replay server ──► FE React: lưới 6 camera + BEV 360° + LiDAR
```

| Phần | File | Vai trò |
|---|---|---|
| Core: hình học | [src/calib.py](src/calib.py), [src/geometry.py](src/geometry.py) | Đọc nội/ngoại tham số Phenikaa (ngoại tham số là LiDAR → camera, nên `T_cam2ego = inv(E)`), pixel → mét trên mặt đường |
| Core: dữ liệu | `PhenikaaSource` trong [src/sources.py](src/sources.py) | Đồng bộ 6 camera theo timestamp (≤ 50 ms), ego pose, nhãn LiDAR |
| Core: detection | [src/detect.py](src/detect.py), [src/detect_cache.py](src/detect_cache.py) | YOLO trên Colab, ghi `cache/<dataset>/<scene>/<model>.jsonl`, đo M3 |
| Core: đánh giá | [src/eval_bev.py](src/eval_bev.py), [src/eval_zones.py](src/eval_zones.py), [tools/find_cases.py](tools/find_cases.py) | M2, M4, tìm khung cho ca lỗi L1–L4 |
| BE | [src/pipeline.py](src/pipeline.py), [src/fuse.py](src/fuse.py), [src/tracker.py](src/tracker.py), [src/zones.py](src/zones.py), [src/app.py](src/app.py) | Gộp, bám vật, vùng cảnh báo, log CSV; chi tiết ở [docs/BE.md](docs/BE.md) |
| FE | [frontend/](frontend/) | Dashboard React; nguồn Replay / Demo / Backend / JSON / Webcam |
| Nối BE → FE | [src/replay_server.py](src/replay_server.py), [src/web_render.py](src/web_render.py) | Phục vụ từng khung và ảnh camera cho web; luồng trực tiếp qua `POST /state` |

Hệ tọa độ xe: x hướng trước, y sang trái, z hướng lên, đơn vị mét.

## Kết quả đo (YOLO11m)

Số đo trên dữ liệu ghi sẵn; chưa đo độ trễ end-to-end trên xe thật.

**M2: sai số vị trí BEV so với nhãn LiDAR** (trung vị / p90, mét; ghép detection với nhãn trên ảnh, IoU ≥ 0,5)

| Khoảng cách | Phenikaa (2 scene) | nuScenes (0553, 0796, 1094) |
|---|---|---|
| 0–5 m | 0,15 / 0,47 | không có cặp |
| 5–10 m | 0,54 / 1,11 | 1,13 / 1,68 |
| 10–20 m | 1,63 / 4,48 | 1,74 / 4,23 |
| 20–30 m | 3,80 / 9,38 | 4,96 / 18,0 |
| > 30 m | 9,81 / 34,2 | 11,4 / 352 |

Sai số gần như chỉ nằm dọc tia nhìn và tăng theo bình phương khoảng cách: σ xuyên tâm ≈ 0,1 + 0,009·d² m, σ ngang 0,07–0,25 m. Trên 30 m hệ thống bật KHÔNG CHẮC và không cảnh báo đỏ.

**M3: thời gian suy luận trên Colab G4** (một khung 6 camera, imgsz 1280): Phenikaa 47,7 ms p50 / 50,2 ms p95 (≈ 21 khung/s); nuScenes 34 ms p50 (≈ 30 khung/s). Chỉ tính suy luận YOLO.

**M4: cảnh báo SÁT SƯỜN theo khung**, Phenikaa 400 khung: precision 0,73, recall 0,86, F1 0,79.

**Tune tracker theo dữ liệu** ([tools/tune_tracker.py](tools/tune_tracker.py), cấu hình trong [config/phenikaa.json](config/phenikaa.json)): nhiễu đo theo khoảng cách và hướng lấy từ M2, cổng ghép Mahalanobis. So với mặc định: số track mới mỗi khung giảm từ 14,2 xuống 6,7, track ma từ 27,6 xuống 22,7, độ phủ giữ 0,86, M4 F1 từ 0,77 lên 0,79. Tune và đo trên cùng 2 scene.

## Giới hạn

- Giả định đường phẳng; dốc, gờ giảm tốc, xe nghiêng làm lệch điểm tiếp đất.
- Với ô tô, đáy box 2D là phần gần nhất của xe, còn đáp án là tâm box 3D, nên sai số ô tô lớn hơn (trung vị khoảng 4,6 m).
- Phần lớn vật bị gắn cờ KHÔNG CHẮC (confidence < 0,55, xa > 30 m, track chỉ dự đoán); chưa có ngưỡng đỏ cho SÁT SƯỜN và TẠT ĐẦU.
- Phenikaa chỉ có 2 scene × 20 s; nhãn không có visibility và track ID, nên TẠT ĐẦU và PHÍA TRƯỚC chưa có số đo tự động.
- YOLO COCO chưa fine-tune; ban đêm (nuScenes scene-1094) yếu hơn.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v       # 39 test: BE, hình học, cache, đánh giá, tracker
cd frontend; npm run typecheck; npm test; npm run build
```

## Colab

Chạy cache YOLO và đo M3 trên GPU Colab: [docs/COLAB_RUN.md](docs/COLAB_RUN.md). Fine-tune nhẹ (đóng băng backbone, chưa chạy): [docs/FINETUNE.md](docs/FINETUNE.md).

## Dữ liệu

`data/`, `cache/` và video không nằm trong repo.

- **nuScenes mini** (CC BY-NC-SA 4.0): tải từ https://www.nuscenes.org/nuscenes#download, giải nén vào `data/mini-nuScenes/`. Chỉ dùng scene-0553, 0796, 1094.
- **Phenikaa**: chỉ dùng khi chủ dữ liệu đồng ý; không commit, không lưu lên Drive; chạy `colab/stop.sh` sau khi dùng Colab. Đặt vào `data/phenikaa/`.
- Cách chọn dữ liệu: [data/README.md](data/README.md) (có trong thư mục `data/` cục bộ).

## Tài liệu

- Spec: [SPEC.md](SPEC.md) · Phân công: [docs/PHAN_CONG.md](docs/PHAN_CONG.md)
- Backend và giao kèo tích hợp: [docs/BE.md](docs/BE.md) · FE: [frontend/README.md](frontend/README.md)
- Kế hoạch Core: [docs/CORE_PLAN.md](docs/CORE_PLAN.md)
- Tận dụng từ dự án trước: [docs/REUSE_P115.md](docs/REUSE_P115.md)
