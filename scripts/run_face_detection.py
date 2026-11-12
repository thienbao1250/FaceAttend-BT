"""
scripts/run_face_detection.py
Test face detection SCRFD trên webcam live.
Nhấn Q để thoát.

Dùng: python scripts/run_face_detection.py
"""

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.camera.camera_capture import CameraCapture
from app.vision.face_detector import FaceDetector

logging.basicConfig(
    level=logging.WARNING,   # Tắt log verbose khi demo
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MODEL_PATH = ROOT / "models" / "det_10g.onnx"

# Màu vẽ
BOX_COLOR = (0, 255, 0)      # xanh lá = bbox
LM_COLOR = (0, 120, 255)     # cam = landmark
TEXT_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    """Vẽ bbox, confidence và landmarks lên frame."""
    h, w = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det.as_int_bbox()

        # Giới hạn trong frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Bỏ qua bbox không hợp lệ sau khi clip (không vẽ chấm mồ côi)
        if x2 <= x1 or y2 <= y1:
            continue

        # Vẽ bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Confidence label — đặt bên trong box nếu không đủ chỗ bên trên
        label = f"{det.confidence:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, FONT, 0.6, 1)
        label_y_top = y1 - lh - 6
        label_y_bot = y1
        if label_y_top < 0:          # không đủ chỗ bên trên → đặt bên trong
            label_y_top = y1 + 2
            label_y_bot = y1 + lh + 8
        cv2.rectangle(frame, (x1, label_y_top), (x1 + lw + 4, label_y_bot), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, label_y_bot - 4), FONT, 0.6, (0, 0, 0), 1)

        # Landmarks (5 điểm)
        if det.landmarks is not None:
            for lx, ly in det.landmarks.astype(int):
                lx = int(np.clip(lx, 0, w - 1))
                ly = int(np.clip(ly, 0, h - 1))
                cv2.circle(frame, (lx, ly), 3, LM_COLOR, -1)

    return frame


def run(
    camera_index: int = 0,
    conf: float = 0.75,
    single_mode: bool = True,
    min_face_px: int = 80,
) -> None:
    """
    Tham số:
        camera_index : chỉ số webcam (mặc định 0)
        conf         : conf_threshold cho SCRFD (mặc định 0.75)
        single_mode  : True  → chỉ hiển thị 1 khuôn mặt lớn nhất
                       False → hiển thị tất cả khuôn mặt được detect
        min_face_px  : cạnh tối thiểu (px) của bbox trên frame gốc — lọc ghost nhỏ
                       (mặc định 80px, tương đương khuôn mặt cách ~1.5m với webcam HD)
    """
    if not MODEL_PATH.exists():
        print(f"Model không tìm thấy: {MODEL_PATH}")
        print("Chạy: python scripts/download_models.py")
        sys.exit(1)

    mode_label = "SINGLE (face lớn nhất)" if single_mode else "ALL faces"
    print(f"Tải model: {MODEL_PATH.name} | conf={conf} | min_face={min_face_px}px | chế độ={mode_label}")
    detector = FaceDetector(
        MODEL_PATH,
        conf_threshold=conf,
        nms_threshold=0.3,
        min_face_size=min_face_px,
    )
    print("Model đã tải. Bắt đầu camera...")

    try:
        with CameraCapture(camera_index=camera_index) as cam:
            info = cam.get_info()
            print(f"Camera: {info['width']}x{info['height']} @ {info['fps']:.0f} FPS")
            print("Nhấn Q để thoát | S để toggle single/all mode.")

            start = time.time()
            frame_count = 0

            while True:
                frame = cam.read_frame()
                frame_count += 1

                # Phát hiện khuôn mặt
                all_detections = detector.detect(frame)

                if single_mode:
                    best = FaceDetector.largest_face(all_detections)
                    shown = [best] if best is not None else []
                else:
                    shown = all_detections

                # Vẽ kết quả
                frame = draw_detections(frame, shown)

                # Hiển thị FPS và số face
                elapsed = time.time() - start
                fps = frame_count / elapsed if elapsed > 0 else 0
                face_info = f"Face: {len(shown)}" + (
                    f"/{len(all_detections)} (single mode)" if single_mode else ""
                )
                info_text = f"FPS: {fps:.1f} | {face_info}"
                cv2.putText(frame, info_text, (10, 30), FONT, 0.9, (255, 255, 0), 2)

                cv2.imshow("FaceAttend - Face Detection (SCRFD)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    single_mode = not single_mode
                    mode_label = "SINGLE" if single_mode else "ALL"
                    print(f"Chuyển chế độ: {mode_label}")

    except RuntimeError as e:
        print(f"Lỗi: {e}")
        sys.exit(1)

    total_time = time.time() - start
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nTổng kết: {frame_count} frames | {total_time:.1f}s | FPS trung bình: {avg_fps:.1f}")


if __name__ == "__main__":
    # conf=0.5 là ngưỡng gốc đã hoạt động tốt với det_10g.onnx
    # min_face_px=0 để TẮT filter kích thước (không lọc gì)
    # single_mode=True để chỉ hiển thị 1 face lớn nhất
    run(camera_index=0, conf=0.5, single_mode=True, min_face_px=0)

