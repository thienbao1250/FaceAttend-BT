# FaceAttend-BT

Hệ thống điểm danh nhận diện khuôn mặt — phát triển theo từng giai đoạn.

## Yêu cầu

- Python 3.12+
- Webcam hoặc camera IP

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy thử camera

```powershell
python scripts/run_camera_preview.py
```

Nhấn **Q** trong cửa sổ video để thoát.

## Chạy unit test

```powershell
python -m pytest tests/ -v
```

## Cấu trúc dự án

```
app/
  camera/         ← Module camera (Giai đoạn 1) ✅
  vision/         ← Preprocessing, detection, alignment, embedding (Giai đoạn 2–5)
tests/
  unit/
scripts/          ← Script chạy thử
models/           ← Model weights (không commit lên Git)
docs/
requirements.txt
```

## Tiến độ

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 0 | Khảo sát repository | ✅ Hoàn thành |
| 1 | Camera module | ✅ Hoàn thành |
| 2 | Frame preprocessing | 🔜 Tiếp theo |
| 3 | SCRFD face detection | ⏳ Chưa thực hiện |
| 4 | Face alignment & quality | ⏳ Chưa thực hiện |
| 5 | ArcFace embedding | ⏳ Chưa thực hiện |
| 6–16 | ... | ⏳ Chưa thực hiện |
