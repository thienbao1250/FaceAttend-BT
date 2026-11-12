"""
scripts/run_attendance.py
Demo Giai đoạn 8 — Pipeline điểm danh đầy đủ.

Camera → SCRFD detect → align → quality → ArcFace embed
→ FaceMatcher → AttendanceService → HUD + lịch sử

Nạp gallery từ file data/gallery.npz (đã đăng ký qua run_enrollment.py).
Lịch sử sự kiện xuất ra data/attendance_log.json khi thoát.

Phím:
  Q / ESC   : thoát và xuất log
  L         : in lịch sử ra terminal
  R         : reset state cooldown (bắt đầu lại vòng IN/OUT)

Chạy: .venv\\Scripts\\python.exe scripts/run_attendance.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.camera.camera_capture import CameraCapture
from app.vision.face_detector import FaceDetector
from app.vision.face_alignment import align_face, crop_face_simple, ALIGNED_SIZE
from app.vision.face_quality import FaceQualityChecker, QualityThresholds
from app.vision.face_recognizer import FaceRecognizer
from app.vision.face_matcher import FaceGallery, FaceMatcher, MatchStatus, MatchThresholds
from app.attendance.attendance_service import (
    AttendanceService,
    EventType,
    SkipReason,
    DEFAULT_TIMEZONE,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
DETECTOR_PATH    = ROOT / "models" / "det_10g.onnx"
RECOGNIZER_PATH  = ROOT / "models" / "w600k_r50.onnx"
GALLERY_FILE     = ROOT / "data" / "gallery.npz"
ATTENDANCE_LOG   = ROOT / "data" / "attendance_log.json"

# ---------------------------------------------------------------------------
# Tham số nghiệp vụ điểm danh
# ---------------------------------------------------------------------------
COOLDOWN_SECONDS    = 30.0    # Giây giữa 2 sự kiện cùng người/cùng camera
MIN_TOGGLE_SECONDS  = 60.0    # Giây tối thiểu giữa CHECK_IN → CHECK_OUT
CAMERA_ID           = "cam_0"

# ---------------------------------------------------------------------------
# Màu
# ---------------------------------------------------------------------------
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
YELLOW = (0, 200, 220)
CYAN   = (220, 220, 0)
WHITE  = (220, 220, 220)
ORANGE = (0, 140, 255)
GRAY   = (140, 140, 140)
FONT   = cv2.FONT_HERSHEY_SIMPLEX


def _put(img, text, pos, color=WHITE, scale=0.5, thick=1):
    cv2.putText(img, text, pos, FONT, scale, color, thick, cv2.LINE_AA)


def _draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    colors = [(255, 100, 100), (100, 100, 255), (100, 255, 100),
              (255, 255, 100), (100, 255, 255)]
    for i, (lx, ly) in enumerate(landmarks.astype(int)):
        lx, ly = int(np.clip(lx, 0, w - 1)), int(np.clip(ly, 0, h - 1))
        cv2.circle(frame, (lx, ly), 3, colors[i], -1)


def _status_color(status):
    if status == MatchStatus.KNOWN:   return GREEN
    if status == MatchStatus.UNCERTAIN: return YELLOW
    return RED


def _now_local() -> str:
    return datetime.now(DEFAULT_TIMEZONE).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Vòng lặp chính
# ---------------------------------------------------------------------------

def run(camera_index: int = 0, conf: float = 0.5) -> None:
    # Kiểm tra model
    for p in [DETECTOR_PATH, RECOGNIZER_PATH]:
        if not p.exists():
            print(f"[LỖI] Model không tìm thấy: {p}")
            print("Chạy: .venv\\Scripts\\python.exe scripts/download_models.py")
            sys.exit(1)

    print(f"Tải detector  : {DETECTOR_PATH.name} ...")
    detector = FaceDetector(DETECTOR_PATH, conf_threshold=conf,
                            nms_threshold=0.3, min_face_size=40)

    print(f"Tải recognizer: {RECOGNIZER_PATH.name} ...")
    recognizer = FaceRecognizer(RECOGNIZER_PATH)

    checker = FaceQualityChecker(QualityThresholds(
        min_face_size_px=50,
        min_sharpness=35.0,
        min_brightness=25.0,
        max_brightness=235.0,
        max_yaw_deg=35.0,
        min_confidence=0.45,
    ))

    # Nạp gallery
    gallery = FaceGallery.load(GALLERY_FILE)
    if gallery.size == 0:
        print(
            "[CẢNH BÁO] Gallery rỗng! Chạy run_enrollment.py trước để đăng ký nhân viên.\n"
            "           Hệ thống vẫn chạy nhưng mọi khuôn mặt đều là UNKNOWN."
        )
    else:
        print(f"Gallery: {gallery.size} nhân viên | {gallery.total_samples} mẫu")

    thresholds = MatchThresholds(accept=0.35, reject=0.20, margin_min=0.07)
    matcher    = FaceMatcher(gallery, thresholds)

    attendance = AttendanceService(
        cooldown_seconds=COOLDOWN_SECONDS,
        min_toggle_seconds=MIN_TOGGLE_SECONDS,
    )

    print(
        f"\nSẵn sàng! Cooldown={COOLDOWN_SECONDS}s | MinToggle={MIN_TOGGLE_SECONDS}s\n"
        "  [Q/ESC] = thoát & xuất log  [L] = in lịch sử  [R] = reset state\n"
    )

    PREVIEW_SIZE    = ALIGNED_SIZE * 2    # 224×224
    blank_preview   = np.zeros((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8)

    # Thông điệp phản hồi tạm thời (hiển thị 2.5s)
    flash_msg: str           = ""
    flash_color              = WHITE
    flash_until: float       = 0.0

    # Bảng sự kiện ngắn gọn hiển thị trên màn hình (giữ 5 dòng cuối)
    recent_events: list[str] = []

    start_time  = time.time()
    frame_count = 0

    try:
        with CameraCapture(camera_index=camera_index) as cam:
            cam_info     = cam.get_info()
            cam_h, cam_w = cam_info["height"], cam_info["width"]
            print(f"Camera: {cam_w}×{cam_h}")

            while True:
                frame = cam.read_frame()
                frame_count += 1
                display = frame.copy()

                # --- AI pipeline ---
                try:
                    detections = detector.detect(frame)
                except Exception:
                    detections = []

                best          = FaceDetector.largest_face(detections)
                aligned_img   = None
                match_result  = None
                quality_ok    = False

                if best is not None:
                    # Align
                    try:
                        if best.landmarks is not None:
                            aligned_img = align_face(frame, best.landmarks).image
                        else:
                            aligned_img = crop_face_simple(frame, best.bbox)
                    except Exception:
                        aligned_img = None

                    if aligned_img is not None:
                        # Quality check
                        qr = checker.check(
                            aligned_img,
                            confidence=best.confidence,
                            face_bbox=best.bbox,
                            landmarks=best.landmarks,
                        )
                        quality_ok = qr.passed

                        if quality_ok:
                            # Embedding + match
                            try:
                                emb          = recognizer.get_embedding(aligned_img)
                                match_result = matcher.match(emb)
                            except Exception:
                                match_result = None
                        else:
                            # Hiển thị hướng dẫn chất lượng
                            x1, y1, x2, y2 = best.as_int_bbox()
                            x1c = max(0, x1); y1c = max(0, y1)
                            x2c = min(cam_w - 1, x2); y2c = min(cam_h - 1, y2)
                            if x2c > x1c and y2c > y1c:
                                cv2.rectangle(display, (x1c, y1c), (x2c, y2c), RED, 2)
                            _put(display, qr.message,
                                 (x1c, min(y2c + 18, cam_h - 6)), RED, 0.4)

                    # --- Vẽ bounding box chính ---
                    if best is not None and match_result is not None:
                        x1, y1, x2, y2 = best.as_int_bbox()
                        x1c = max(0, x1); y1c = max(0, y1)
                        x2c = min(cam_w - 1, x2); y2c = min(cam_h - 1, y2)
                        box_color = _status_color(match_result.status)
                        if x2c > x1c and y2c > y1c:
                            cv2.rectangle(display, (x1c, y1c), (x2c, y2c), box_color, 2)
                        if best.landmarks is not None:
                            _draw_landmarks(display, best.landmarks)

                        # Nhãn tên nhân viên
                        label = (
                            match_result.employee_id
                            if match_result.employee_id
                            else match_result.status.value.upper()
                        )
                        sim_txt = f"{match_result.similarity:.2f}"
                        _put(display, f"{label}  {sim_txt}",
                             (x1c + 2, max(y1c - 8, 14)), box_color, 0.55, 2)

                        # --- Chấm công ---
                        decision = attendance.process(match_result, camera_id=CAMERA_ID)
                        if decision.created:
                            ev = decision.event
                            icon  = "IN " if ev.event_type == EventType.CHECK_IN else "OUT"
                            line  = f"[{icon}] {ev.employee_id}  {_now_local()}"
                            flash_msg    = line
                            flash_color  = GREEN if ev.event_type == EventType.CHECK_IN else CYAN
                            flash_until  = time.time() + 2.5
                            recent_events.append(line)
                            if len(recent_events) > 5:
                                recent_events.pop(0)
                            print(f"  {ev}")
                        else:
                            # Hiển thị cooldown bar nhỏ dưới bbox (nếu có nhân viên đã biết)
                            if (decision.skip_reason in (SkipReason.COOLDOWN, SkipReason.MIN_TOGGLE)
                                    and match_result.employee_id):
                                remain = decision.cooldown_remaining
                                bar_w  = x2c - x1c
                                filled = int(bar_w * max(0, 1 - remain / COOLDOWN_SECONDS))
                                bar_y  = min(y2c + 4, cam_h - 6)
                                cv2.rectangle(display, (x1c, bar_y), (x2c, bar_y + 4), (60, 60, 60), -1)
                                if filled > 0:
                                    cv2.rectangle(display, (x1c, bar_y), (x1c + filled, bar_y + 4), ORANGE, -1)

                # --- HUD góc trái trên ---
                _put(display, f"Gallery: {gallery.size} NV | {attendance.total_events} su kien",
                     (10, 22), CYAN, 0.5, 1)
                _put(display, f"Faces: {len(detections)}",
                     (10, 42), WHITE, 0.42)

                # --- Flash message ---
                if time.time() < flash_until and flash_msg:
                    _put(display, flash_msg, (10, cam_h - 50), flash_color, 0.65, 2)

                # --- Danh sách sự kiện gần nhất (góc phải) ---
                for i, line in enumerate(reversed(recent_events)):
                    _put(display, line,
                         (cam_w - 240, cam_h - 20 - i * 20), WHITE, 0.38)

                # --- Phím tắt góc dưới ---
                _put(display, "[Q]=thoat  [L]=lich su  [R]=reset",
                     (10, cam_h - 14), GRAY, 0.38)

                # --- FPS ---
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                _put(display, f"FPS:{fps:.1f}", (cam_w - 70, 20), GRAY, 0.4)

                cv2.imshow("FaceAttend - Stage 8: Attendance", display)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                elif key == ord("l"):
                    attendance.print_summary()
                elif key == ord("r"):
                    attendance.reset_state()
                    print("  [R] Đã reset toàn bộ trạng thái cooldown/tracking.")

    except RuntimeError as exc:
        print(f"\n[LỖI CAMERA] {exc}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()

    # --- Xuất log khi thoát ---
    total_time = time.time() - start_time
    avg_fps    = frame_count / total_time if total_time > 0 else 0

    attendance.print_summary()
    attendance.export_json(str(ATTENDANCE_LOG))
    attendance.export_csv(str(ATTENDANCE_LOG.with_suffix(".csv")))

    print(f"  Thời gian: {total_time:.1f}s  |  FPS TB: {avg_fps:.1f}")
    print(f"  Đã xuất log → {ATTENDANCE_LOG.relative_to(ROOT)}")
    print(f"             → {ATTENDANCE_LOG.with_suffix('.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    run(camera_index=0, conf=0.5)

