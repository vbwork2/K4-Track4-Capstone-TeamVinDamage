# Fine-tune nhẹ YOLO trên Colab (việc của Core)

Đi kèm [SPEC.md](../SPEC.md) bản 2. Đây là phần **thêm**: chỉ làm khi bản chính (Replay Phenikaa + Live) đã chạy được lúc 1:15. Fine-tune hỏng thì vẫn dùng `yolo11n` / `yolo11m` COCO như spec.

## 1. Làm gì và vì sao

- YOLO COCO bắt xe máy nhỏ, bị che, ban đêm kém (L1, L3). Fine-tune nhẹ trên ảnh xe máy ở góc camera xe có thể giúp.
- **Không dùng LoRA đúng nghĩa.** LoRA là kỹ thuật cho transformer; Ultralytics không hỗ trợ LoRA cho YOLO (CNN). Cách tương đương là **đóng băng backbone** (`freeze=10`, layer 0–9 của yolo11) và chỉ train neck + head: ít tham số, nhanh, ít quên COCO. Khi giám khảo hỏi thì nói đúng như vậy.
- Giữ nguyên id lớp COCO (person 0, bicycle 1, car 2, motorcycle 3, bus 5, truck 7):
  - `detect.py` và cache không phải đổi gì, chỉ thêm một tên model `yolo11n_ft`;
  - so được model gốc với model fine-tune trên **cùng tập test**.

## 2. Dữ liệu: `data/finetune_nusc/` (422 MB, lên Drive được)

Tạo bằng `python -I tools/nusc_to_yolo.py <P-115>/data/mini-nuScenes data/finetune_nusc` (cần nuscenes-devkit).

| Split | Scene nuScenes mini | Ảnh (6 camera × keyframe) | Box |
|---|---|---|---|
| train | 0103, 0655, 0757, 0916, 1077 (đêm), 1100 (đêm) | 1.464 | person 1.646, car 3.726, motorcycle 283, bicycle 59, bus 257, truck 238 |
| val | 0061 (đủ 6 lớp, dùng để dừng sớm) | 234 | person 958, car 289, motorcycle 52, bicycle 55, bus 24, truck 140 |
| test | **0553, 0796, 1094**: đúng 3 scene của M2/M4 | 726 | person 1.292, car 1.256, motorcycle 35, bicycle 74, bus 132, truck 96 |

- **Không bao giờ train trên test.** 3 scene test chính là dữ liệu đo M2/M4, train lên đó thì số đo vô nghĩa.
- Nhãn 2D = khung bao của 8 góc box 3D chiếu vào ảnh. Lọc: `visibility` ≥ 2, ≥ 50% box nằm trong ảnh (giống luật M2), cao ≥ 8 px, mọi góc nằm trước camera.
- Box motorcycle/bicycle của nuScenes đã gồm người lái, nên model học ra box "rider" luôn. Hệ quả cho `fuse.py` (báo BE): với model fine-tune, box motorcycle đứng một mình cũng là rider; ít khi có box person chồng lên.
- Box chiếu từ 3D hơi rộng hơn vật thật (nhất là xe ngang). Đây là giới hạn của nhãn, ghi vào slide.
- Đã kiểm bằng mắt: vẽ nhãn lên ảnh train ngày/đêm, val, test, thấy box khớp vật.
- Xe máy ít (283 box train). Đây là điểm yếu lớn nhất của bộ này, xem mục 6.

## 3. Đưa dữ liệu lên Colab

Cách A, từ Drive (thư mục chia sẻ):
1. Trong terminal WSL thật (không qua công cụ tự động, vì `drivemount` cần bấm link): gắn Drive vào session.
2. **Chép về đĩa VM rồi mới train.** Đọc trực tiếp từ Drive qua FUSE rất chậm:
   `cp -r "/content/drive/MyDrive/<thư mục chia sẻ>/data/finetune_nusc" /content/`

Cách B, không cần Drive (khuyên dùng lúc thi, ít bước tay):
`tar cf ft.tar -C data finetune_nusc` (không nén vì đã là JPEG), rồi `colab upload` lên `/content/` và `tar xf`. Có thể thêm biến `WITH_FT=1` vào `up.sh` giống `WITH_PHENIKAA`.

`data.yaml` đã để `path: /content/finetune_nusc`.

## 4. Chạy

```
# trên Colab, trong /content/hack (cần chép thêm tools/ft_yolo.py lên)
python tools/ft_yolo.py --data /content/finetune_nusc/data.yaml --model yolo11n.pt --out results/
# nếu còn giờ:
python tools/ft_yolo.py --data /content/finetune_nusc/data.yaml --model yolo11m.pt --batch 8 --out results/
```

Chạy qua `colab/run.sh` (đặt `JOB=...`) để chạy nền, xem bằng `log.sh`. Script làm 3 việc:
1. đo `yolo11n` COCO gốc trên test;
2. train 30 epoch, imgsz 1280, `freeze=10`, AdamW lr 0,001, dừng sớm nếu 10 epoch không tiến bộ;
3. đo lại bản fine-tune trên test, ghi `results/ft_yolo11n.json` (mAP50, mAP50-95, AP50 theo lớp, phút train) và `results/yolo11n_ft.pt`.

Chưa chạy thử trên Colab, nên chưa có số thời gian. Chưa đo thì ghi "chưa đo".

## 5. Lịch trong 2 giờ (Core)

| Giờ | Việc fine-tune | Song song |
|---|---|---|
| 0:00–0:15 | `up.sh` kèm `finetune_nusc` + `tools/ft_yolo.py` | khung repo |
| 0:15–0:20 | bắt đầu `ft_yolo.py` yolo11n chạy nền | cache `yolo11n` Phenikaa cùng GPU (nếu chậm thì cache trước, train sau) |
| 0:20–1:15 | **không đụng tới**, chỉ xem log thỉnh thoảng | calib, kiểm ngoại tham số, `geometry.py`, `PhenikaaSource`, ghép lúc 1:15 |
| 1:15–1:30 | đọc `ft_yolo11n.json`; nếu mAP test tăng: chạy `detect_cache` với `yolo11n_ft.pt` cho Phenikaa + nuScenes | M2 bản COCO |
| 1:30–2:00 | M2/M4 thêm cột `yolo11n_ft`; tải `.pt` về local để thử Live | M3, M4, L3, video V1 |
| sau 2:00 | đóng băng; `stop.sh` | |

Cắt giảm theo thứ tự: bỏ yolo11m fine-tune → bỏ chạy Live bằng model fine-tune → bỏ toàn bộ fine-tune. Không vì fine-tune mà trễ mốc 1:15.

## 6. Báo cáo kết quả (trung thực)

- Bảng: mAP50 và AP50 motorcycle/person trên test, COCO gốc vs fine-tune; M2 theo dải khoảng cách cho 3 model.
- Nói rõ:
  - train trên nuScenes (Boston/Singapore), **test cùng miền**; lên Phenikaa (Hà Nội) có thể không tăng, M2 Phenikaa sẽ cho thấy;
  - xe máy chỉ 283 box train, 35 box test → AP xe máy dao động mạnh, đừng kết luận từ một con số;
  - dữ liệu đã chuẩn bị trước buổi; phần train chạy trong giờ build.
- Sửa SPEC khi quyết định dùng (báo cả nhóm): §8 dòng "Học" (đổi "không tự huấn luyện" thành "fine-tune nhẹ, đóng băng backbone"), §3 thêm `yolo11n_ft`, §6 M2 thêm cột.

## 7. Phenikaa (sau, không làm trong 2 giờ)

Phenikaa là đường Việt Nam, nhiều xe máy (khoảng 28.000 nhãn Rider). Đây mới là dữ liệu fine-tune đáng giá, nhưng:
- phải kiểm xong chiều ngoại tham số (chiếu nhãn vào `CAM_P_F`, so với `COMPLEX_CITY_projected_3d_box_5x2.mp4`) trước khi sinh nhãn 2D;
- cần chủ dữ liệu đồng ý cho đưa lên Colab; **không để lên Drive**, đưa thẳng bằng `colab upload`;
- chia: train COMPLEX URBAN, test COMPLEX CITY (scene demo); không có visibility nên lọc bằng ≥ 50% box trong ảnh và bỏ vật xa > 50 m;
- nhãn `Rider` → id 3 (motorcycle), `Pedestrian` → 0, `Car` → 2;
- để ở `data/finetune_phenikaa/`, ghi "Không lên Drive" trong `data/README.md`.
