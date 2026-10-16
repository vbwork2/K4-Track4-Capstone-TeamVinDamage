# VinDamage — Feature-only Frontend

**Chỉ có một trang tính năng `http://localhost:5173/`. Không có đăng nhập, đăng ký, landing page, dashboard P-115 hoặc backend auth.** Được tách từ màn hình VinDamage React trong FE P-115; chỉ giữ lại những component cần thiết cho cảnh báo đa camera.

## Chạy local

Yêu cầu Node.js 20.19+ hoặc 22.12+ và Python 3.10+ (chỉ cần Python nếu muốn stream dữ liệu).

```bash
npm install
npm run dev
```

Truy cập **http://localhost:5173/**, mở thẳng trang giám sát. Mặc định là **MOCK DATA** (ảnh camera không giả mạo, chỉ có track và bounding box mô phỏng).

## Các tính năng có sẵn

- Lưới camera 2×3: 6 camera Phenikaa hoặc camera nuScenes, nhận JPEG/PNG URL/data URL từ Backend; vẽ bbox với cảnh báo.
- BEV 360°: track, trail, vận tốc, vùng cảnh báo; ground-truth và layer LiDAR tùy chọn (tọa độ ego x về trước, y sang trái, mét).
- Status cảnh báo: thường / cảnh báo / nguy hiểm / **không chắc** (không bao giờ tô đỏ đối tượng `uncertain: true`).
- Thanh chọn nguồn: **Demo**, **Backend**, **Nhập JSON**, **Webcam**; menu chọn 4 kịch bản giả lập, timeline, pause/step, video WebM bằng tab capture.
- FPS/latency là số liệu Backend; mock không giả số hiệu năng; webcam chỉ hiển thị preview, chưa chạy inference.

## Kết nối Backend Python

Terminal thứ hai:

```bash
python tools/vindamage_bridge.py
```

Gửi JSON bằng `POST http://127.0.0.1:8765/state`, FE đọc `GET /warning-api/state` qua Vite proxy. Nhấn **Backend** trên thanh chức năng.

Ví dụ state (có thể lưu thành `sample.json`, sau đó chọn **Nhập JSON**):

```json
{
  "t": 1700000000,
  "source": "phenikaa",
  "scene": "test",
  "model": "YOLO",
  "images": {},
  "dets": [],
  "tracks": [{"id": 12, "cls": "motorcycle", "x": 3.0, "y": -2.1, "vx": -1, "vy": 0, "trail": [[3.7,-2.1],[3.0,-2.1]], "zone": "SAT_SUON", "level": 2, "ttc": 1.1, "uncertain": false}],
  "gt": [],
  "stats": {"fps": 10.5, "latency_ms": 92, "n_tracks": 1, "alerts": {"SAT_SUON": 1, "TAT_DAU": 0, "PHIA_TRUOC": 0}}
}
```

Đối với ảnh, thêm `images` với mỗi camera là URL truy cập từ trình duyệt hoặc chuỗi `data:image/jpeg;base64,...` và `image_sizes` `[width,height]`. Tên Phenikaa: `CAM_P_F`, `CAM_P_FL`, `CAM_P_FR`, `CAM_P_L`, `CAM_P_R`, `CAM_P_B`.

**Lưu ý:** bridge chỉ dành cho mạng local tin cậy, không có xác thực nên không nên publish trực tiếp ra Internet. Backend quyết định TTC/level; FE chỉ vẽ. Xuất browser recording là `.webm` (không phải `.mp4`); nếu phải có MP4 chuẩn dùng bản FE OpenCV.

## Kiểm thử

```bash
npm run typecheck
npm test
npm run build
```

Các file FE gọn trong `src/warning/` cùng Vite config, không phụ thuộc API hay tài khoản P-115. Trang index mở trực tiếp, không redirect.
