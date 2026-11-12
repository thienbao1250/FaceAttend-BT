"""
scripts/debug_detection.py
Chạy detect KHÔNG lọc gì (conf=0.3, min_size=0) để in raw output từ model.
Dùng để tìm ngưỡng phù hợp.

Chạy: python scripts/debug_detection.py
"""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.camera.camera_capture import CameraCapture
from app.vision.face_detector import FaceDetector

MODEL_PATH = ROOT / "models" / "det_10g.onnx"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Chạy detect với ngưỡng rất thấp để xem tất cả output
detector = FaceDetector(MODEL_PATH, conf_threshold=0.3, nms_threshold=0.4, min_face_size=0)

print("DEBUG mode: conf=0.3, min_size=0 — hiển thị tất cả detection")
print("Nhấn Q để thoát")

with CameraCapture(camera_index=0) as cam:
    while True:
        frame = cam.read_frame()
        h, w = frame.shape[:2]

        detections = detector.detect(frame)

        # Vẽ TẤT CẢ detection với màu theo kích thước
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            bw = x2 - x1
            bh = y2 - y1

            # Màu theo kích thước: nhỏ=đỏ, trung=vàng, lớn=xanh
            if min(bw, bh) < 40:
                color = (0, 0, 255)   # đỏ = rất nhỏ (ghost)
            elif min(bw, bh) < 80:
                color = (0, 165, 255) # cam = nhỏ
            else:
                color = (0, 255, 0)   # xanh = OK

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"c={det.confidence:.2f} {bw}x{bh}px"
            cv2.putText(frame, label, (x1, max(y1 - 4, 12)), FONT, 0.45, color, 1)

        # Header info
        info = f"Total detections: {len(detections)} | cam {w}x{h}"
        cv2.putText(frame, info, (10, 25), FONT, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "RED<40px  ORANGE<80px  GREEN>=80px", (10, h - 10), FONT, 0.5, (200, 200, 200), 1)

        cv2.imshow("DEBUG Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()

