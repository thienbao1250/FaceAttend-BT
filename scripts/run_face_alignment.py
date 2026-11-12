"""
scripts/run_face_alignment.py
Test Face Alignment + Quality Check (Giai đoạn 4) trên webcam live.

Hiển thị:
  - Frame gốc bên trái: bbox + landmarks từ SCRFD
  - Crop đã căn chỉnh 112×112 bên phải (phóng to x2 lên 224×224)
  - Kết quả quality check: passed/fail + các chỉ số

Nhấn Q để thoát.
Chạy: python scripts/run_face_alignment.py
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
from app.vision.face_alignment import align_face, crop_face_simple, ALIGNED_SIZE
from app.vision.face_quality import FaceQualityChecker, QualityThresholds

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MODEL_PATH = ROOT / "models" / "det_10g.onnx"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Màu
BOX_COLOR  = (0, 255, 0)
FAIL_COLOR = (0, 0, 255)
LM_COLOR   = (0, 120, 255)

# Kích thước panel hiển thị cho aligned face (phóng to 2×)
PREVIEW_SIZE = ALIGNED_SIZE * 2  # 224×224


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> None:
    """Vẽ 5 landmarks lên frame."""
    h, w = frame.shape[:2]
    colors_lm = [
        (255, 100, 100),  # mắt trái  — xanh nhạt
        (100, 100, 255),  # mắt phải  — đỏ nhạt
        (100, 255, 100),  # mũi       — xanh lá
        (255, 255, 100),  # miệng trái
        (100, 255, 255),  # miệng phải
    ]
    for i, (lx, ly) in enumerate(landmarks.astype(int)):
        lx = int(np.clip(lx, 0, w - 1))
        ly = int(np.clip(ly, 0, h - 1))
        cv2.circle(frame, (lx, ly), 4, colors_lm[i], -1)
        cv2.circle(frame, (lx, ly), 4, (255, 255, 255), 1)


def put_quality_overlay(
    panel: np.ndarray,
    quality,
    x: int = 4,
    y_start: int = 8,
) -> None:
    """Vẽ thông tin quality lên panel aligned face."""
    color = BOX_COLOR if quality.passed else FAIL_COLOR
    status_text = "PASSED" if quality.passed else "FAILED"
    cv2.putText(panel, status_text, (x, y_start + 14), FONT, 0.5, color, 1)

    metrics = [
        f"sharp: {quality.sharpness:.0f}",
        f"bright:{quality.brightness:.0f}",
        f"yaw:   {quality.yaw_deg:+.1f}°",
        f"conf:  {quality.confidence:.2f}",
    ]
    for i, m in enumerate(metrics):
        cv2.putText(panel, m, (x, y_start + 32 + i * 16), FONT, 0.4, (220, 220, 220), 1)

    if not quality.passed:
        # Hiển thị lý do fail (xuống dưới)
        for j, reason in enumerate(quality.fail_reasons):
            cv2.putText(
                panel,
                f"! {reason.name}",
                (x, y_start + 32 + len(metrics) * 16 + j * 14),
                FONT, 0.38, FAIL_COLOR, 1,
            )


def run(camera_index: int = 0, conf: float = 0.5) -> None:
    if not MODEL_PATH.exists():
        print(f"Model không tìm thấy: {MODEL_PATH}")
        print("Chạy: python scripts/download_models.py")
        sys.exit(1)

    print(f"Tải model: {MODEL_PATH.name} ...")
    detector = FaceDetector(MODEL_PATH, conf_threshold=conf, nms_threshold=0.3, min_face_size=30)
    checker  = FaceQualityChecker(QualityThresholds(
        min_face_size_px=50,
        min_sharpness=35.0,
        min_brightness=30.0,
        max_brightness=230.0,
        max_yaw_deg=35.0,
        min_confidence=0.45,
    ))
    print("Model đã tải. Bắt đầu camera... | Nhấn Q để thoát.")

    # Canvas hiển thị: frame gốc (trái) + aligned preview (phải, 224×224) ghép ngang
    blank_preview = np.zeros((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8)

    try:
        with CameraCapture(camera_index=camera_index) as cam:
            cam_info = cam.get_info()
            cam_h = cam_info["height"]
            cam_w = cam_info["width"]
            print(f"Camera: {cam_w}x{cam_h}")

            start = time.time()
            frame_count = 0

            while True:
                frame = cam.read_frame()
                frame_count += 1

                # Detect
                detections = detector.detect(frame)

                # Chọn 1 face lớn nhất (giống single_mode)
                best = FaceDetector.largest_face(detections)

                # Tạo preview panel
                preview = blank_preview.copy()
                quality_result = None

                if best is not None:
                    x1, y1, x2, y2 = best.as_int_bbox()
                    # Clip bbox
                    h_f, w_f = frame.shape[:2]
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w_f - 1, x2), min(h_f - 1, y2)

                    # Alignment
                    if best.landmarks is not None:
                        aligned = align_face(frame, best.landmarks)
                        aligned_img = aligned.image
                    else:
                        aligned_img = crop_face_simple(frame, best.bbox)

                    # Quality check
                    quality_result = checker.check(
                        aligned_img,
                        confidence=best.confidence,
                        face_bbox=best.bbox,
                        landmarks=best.landmarks,
                    )

                    # Vẽ bbox lên frame gốc
                    box_clr = BOX_COLOR if quality_result.passed else FAIL_COLOR
                    if x2c > x1c and y2c > y1c:
                        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), box_clr, 2)

                    # Vẽ landmarks lên frame gốc
                    if best.landmarks is not None:
                        draw_landmarks(frame, best.landmarks)

                    # Label confidence + quality
                    label = f"{best.confidence:.2f}"
                    cv2.putText(frame, label, (x1c + 2, max(y1c - 6, 12)), FONT, 0.55, box_clr, 1)

                    # Hiển thị thông điệp hướng dẫn
                    if not quality_result.passed:
                        guide_y = min(y2c + 18, h_f - 6)
                        cv2.putText(frame, quality_result.message, (x1c, guide_y),
                                    FONT, 0.42, FAIL_COLOR, 1)

                    # Preview aligned face (phóng x2)
                    preview = cv2.resize(
                        aligned_img,
                        (PREVIEW_SIZE, PREVIEW_SIZE),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    put_quality_overlay(preview, quality_result)

                # Ghép frame + preview (resize frame về cam_h để khớp chiều cao)
                frame_resized = cv2.resize(frame, (cam_w, cam_h))

                # Pad hoặc crop preview theo chiều cao
                if PREVIEW_SIZE < cam_h:
                    pad = cam_h - PREVIEW_SIZE
                    preview = cv2.copyMakeBorder(preview, 0, pad, 0, 0,
                                                  cv2.BORDER_CONSTANT, value=0)
                else:
                    preview = preview[:cam_h]

                canvas = np.hstack([frame_resized, preview])

                # FPS HUD
                elapsed = time.time() - start
                fps = frame_count / elapsed if elapsed > 0 else 0
                face_cnt = len(detections)
                hud = f"FPS: {fps:.1f} | Detected: {face_cnt} | Showing: {'1 (largest)' if best else '0'}"
                cv2.putText(canvas, hud, (10, 24), FONT, 0.7, (255, 255, 0), 2)

                cv2.imshow("FaceAttend - Stage 4: Alignment + Quality", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except RuntimeError as e:
        print(f"Lỗi: {e}")
        sys.exit(1)

    total = time.time() - start
    avg_fps = frame_count / total if total > 0 else 0
    print(f"\nTổng kết: {frame_count} frames | {total:.1f}s | FPS trung bình: {avg_fps:.1f}")


if __name__ == "__main__":
    run(camera_index=0, conf=0.5)

