# VinFast Surround Rider Warning

Cảnh báo xe máy sát sườn, tạt đầu và nguy cơ phía trước bằng 6 camera và BEV 360°. Replay đọc cache Phenikaa hoặc nuScenes; Live dùng một webcam và vùng phía trước. SPEC bản 2 là tài liệu chuẩn.

- Spec: [SPEC.md](SPEC.md)
- Phân công: [docs/PHAN_CONG.md](docs/PHAN_CONG.md)
- Backend, giao kèo tích hợp và cách chạy chưa cần data: [docs/BE.md](docs/BE.md)
- Tận dụng từ dự án trước: [docs/REUSE_P115.md](docs/REUSE_P115.md)
- Colab (tùy chọn, chỉ cho bộ đo M2): [docs/COLAB.md](docs/COLAB.md)

## Kiểm tra BE chưa cần data

```powershell
python -m pip install numpy
python -m unittest discover -s tests -v
python -m src.app --source synthetic --headless --max-frames 60 --log results/log_synthetic.csv --metrics results/m3_local_synthetic.json
```

Dữ liệu và hình học trong chế độ này là giả lập, không dùng làm số đo dự án.

## Chạy với dữ liệu thật sau khi ghép Core và FE
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-be.txt
python -m src.app --source webcam
python -m src.app --source nuscenes --root data/mini-nuScenes --scene scene-1094 --cache cache
```

## Dữ liệu
nuScenes mini (CC BY-NC-SA 4.0) không nằm trong repo. Tải từ https://www.nuscenes.org/nuscenes#download rồi giải nén vào `data/mini-nuScenes/`.
