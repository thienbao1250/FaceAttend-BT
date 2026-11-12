"""
scripts/run_enrollment.py
Demo Giai đoạn 7 — Employee Enrollment qua webcam.

Luồng:
  1. Nhập mã và tên nhân viên từ terminal.
  2. Mở webcam; mỗi frame chạy detect → align → quality → embed → thêm vào pending.
  3. Khi đủ MIN_SAMPLES_REQUIRED mẫu hợp lệ, nhấn [F] để finalize.
  4. Có thể đăng ký nhiều nhân viên liên tiếp trong cùng một phiên.
  5. Cuối phiên in thống kê gallery.

Phím điều khiển trong cửa sổ camera:
  SPACE / S   : chụp frame hiện tại thêm vào pending
  F           : finalize — hoàn tất đăng ký nhân viên hiện tại
  R           : reset pending — bỏ hết mẫu chưa finalize
  N           : đăng ký nhân viên mới (nhập từ terminal)
  Q / ESC     : thoát

Chạy: python scripts/run_enrollment.py
"""

from __future__ import annotations

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
from app.vision.face_quality import FaceQualityChecker, QualityThresholds
from app.vision.face_recognizer import FaceRecognizer
from app.vision.face_matcher import FaceGallery
from app.enrollment.enrollment_service import (
    EnrollmentService,
    SampleStatus,
    EmployeeStatus,
    MIN_SAMPLES_REQUIRED,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Đường dẫn model
# ---------------------------------------------------------------------------
DETECTOR_PATH   = ROOT / "models" / "det_10g.onnx"
RECOGNIZER_PATH = ROOT / "models" / "w600k_r50.onnx"

# ---------------------------------------------------------------------------
# Cấu hình lưu trữ
# ---------------------------------------------------------------------------
SAVE_FACE_IMAGES     = True   # True: tự động lưu ảnh crop ra thư mục; False: chỉ lưu embedding trong RAM
REGISTERED_FACES_DIR = ROOT / "data" / "registered_faces"
GALLERY_FILE_PATH    = ROOT / "data" / "gallery.npz"

# ---------------------------------------------------------------------------
# Màu sắc
# ---------------------------------------------------------------------------
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
YELLOW = (0, 200, 220)
CYAN   = (220, 220, 0)
WHITE  = (220, 220, 220)
ORANGE = (0, 140, 255)
FONT   = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Tiện ích vẽ
# ---------------------------------------------------------------------------

def _put(img: np.ndarray, text: str, pos: tuple, color=WHITE,
         scale: float = 0.5, thickness: int = 1) -> None:
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_progress_bar(img: np.ndarray, x: int, y: int, w: int, h: int,
                       current: int, total: int) -> None:
    """Vẽ thanh tiến trình số mẫu đã thu."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    filled = int(w * min(current, total) / max(total, 1))
    color = GREEN if current >= total else YELLOW
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), WHITE, 1)
    _put(img, f"{current}/{total}", (x + w + 6, y + h - 2), WHITE, 0.4)


def _draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> None:
    colors = [(255, 100, 100), (100, 100, 255), (100, 255, 100),
              (255, 255, 100), (100, 255, 255)]
    h, w = frame.shape[:2]
    for i, (lx, ly) in enumerate(landmarks.astype(int)):
        lx = int(np.clip(lx, 0, w - 1))
        ly = int(np.clip(ly, 0, h - 1))
        cv2.circle(frame, (lx, ly), 3, colors[i], -1)


# ---------------------------------------------------------------------------
# Nhập thông tin nhân viên từ terminal
# ---------------------------------------------------------------------------

def _input_employee_info() -> tuple[str, str, str]:
    """Nhập mã, tên và phòng ban nhân viên từ terminal."""
    print("\n" + "=" * 50)
    print("  ĐĂNG KÝ NHÂN VIÊN MỚI")
    print("=" * 50)
    while True:
        emp_id = input("  Mã nhân viên (vd: EMP001): ").strip()
        if emp_id:
            break
        print("  [!] Mã nhân viên không được để trống.")
    while True:
        name = input("  Họ tên             : ").strip()
        if name:
            break
        print("  [!] Tên không được để trống.")
    dept = input("  Phòng ban (Enter để bỏ qua): ").strip()
    return emp_id, name, dept


# ---------------------------------------------------------------------------
# Vòng lặp chính
# ---------------------------------------------------------------------------

def run(camera_index: int = 0, conf: float = 0.5) -> None:
    # Kiểm tra model
    for p in [DETECTOR_PATH, RECOGNIZER_PATH]:
        if not p.exists():
            print(f"[LỖI] Model không tìm thấy: {p}")
            print("Chạy: python scripts/download_models.py")
            sys.exit(1)

    print(f"Tải detector  : {DETECTOR_PATH.name} ...")
    detector = FaceDetector(DETECTOR_PATH, conf_threshold=conf,
                            nms_threshold=0.3, min_face_size=40)

    print(f"Tải recognizer: {RECOGNIZER_PATH.name} ...")
    recognizer = FaceRecognizer(RECOGNIZER_PATH)

    gallery    = FaceGallery.load(GALLERY_FILE_PATH)
    if gallery.size > 0:
        print(f"  [Gallery] Đã nạp {gallery.size} nhân viên ({gallery.total_samples} mẫu) từ file {GALLERY_FILE_PATH.name}")
    checker    = FaceQualityChecker(QualityThresholds(
        min_face_size_px=60,
        min_sharpness=35.0,
        min_brightness=25.0,
        max_brightness=235.0,
        max_yaw_deg=30.0,
        min_confidence=0.5,
    ))
    service = EnrollmentService(
        detector=detector,
        recognizer=recognizer,
        gallery=gallery,
        quality_checker=checker,
        min_samples=MIN_SAMPLES_REQUIRED,
        max_samples=10,
        min_consistency=0.10,
        save_images_dir=REGISTERED_FACES_DIR if SAVE_FACE_IMAGES else None,
    )

    if SAVE_FACE_IMAGES:
        print(f"  [Chế độ lưu ảnh]: BẬT -> Thư mục: {REGISTERED_FACES_DIR.relative_to(ROOT)}")
    else:
        print("  [Chế độ lưu ảnh]: TẮT -> Chỉ lưu embedding trong RAM")

    # Nhập thông tin nhân viên đầu tiên
    emp_id, name, dept = _input_employee_info()
    try:
        service.create_profile(emp_id, name, department=dept)
        print(f"\n  Hồ sơ tạo thành công: {emp_id} — {name}")
    except ValueError as e:
        print(f"  [LỖI] {e}")
        sys.exit(1)

    current_emp_id = emp_id

    print("\n  Phím: [SPACE/S]=chụp  [F]=finalize  [R]=reset  [N]=nhân viên mới  [Q]=thoát")
    print("  Mở camera ...\n")

    # Trạng thái hiển thị
    last_msg        = ""
    last_msg_color  = WHITE
    last_msg_time   = 0.0

    start_time  = time.time()
    frame_count = 0

    try:
        with CameraCapture(camera_index=camera_index) as cam:
            cam_info = cam.get_info()
            cam_h, cam_w = cam_info["height"], cam_info["width"]
            print(f"  Camera: {cam_w}×{cam_h}")

            while True:
                frame = cam.read_frame()
                frame_count += 1
                display = frame.copy()

                # --- Phát hiện khuôn mặt ---
                try:
                    detections = detector.detect(frame)
                except Exception:
                    detections = []

                best = FaceDetector.largest_face(detections)
                quality_ok = False

                if best is not None:
                    x1, y1, x2, y2 = best.as_int_bbox()
                    h_f, w_f = display.shape[:2]
                    x1c = max(0, x1); y1c = max(0, y1)
                    x2c = min(w_f - 1, x2); y2c = min(h_f - 1, y2)

                    # Kiểm tra quality nhanh để hiển thị màu box
                    from app.vision.face_alignment import align_face, crop_face_simple
                    try:
                        if best.landmarks is not None:
                            aligned = align_face(frame, best.landmarks)
                            aligned_img = aligned.image
                        else:
                            aligned_img = crop_face_simple(frame, best.bbox)

                        qr = checker.check(
                            aligned_img,
                            confidence=best.confidence,
                            face_bbox=best.bbox,
                            landmarks=best.landmarks,
                        )
                        quality_ok = qr.passed
                    except Exception:
                        quality_ok = False

                    box_color = GREEN if quality_ok else RED
                    if x2c > x1c and y2c > y1c:
                        cv2.rectangle(display, (x1c, y1c), (x2c, y2c), box_color, 2)
                    if best.landmarks is not None:
                        _draw_landmarks(display, best.landmarks)
                    _put(display, f"{best.confidence:.2f}",
                         (x1c + 2, max(y1c - 6, 14)), box_color, 0.5)

                    if not quality_ok:
                        try:
                            guide_msg = qr.message
                        except Exception:
                            guide_msg = "Chất lượng thấp"
                        _put(display, guide_msg, (x1c, min(y2c + 18, h_f - 6)), RED, 0.4)

                # --- HUD trên cùng ---
                profile = service.get_profile(current_emp_id)
                pending = service.pending_count(current_emp_id)
                hud_line1 = (
                    f"Dang ky: {current_emp_id} | {profile.name if profile else '?'}"
                )
                _put(display, hud_line1, (10, 22), CYAN, 0.6, 2)

                # Thanh tiến trình
                _put(display, "Mau:", (10, 48), WHITE, 0.42)
                _draw_progress_bar(display, 55, 36, 160, 16,
                                   pending, MIN_SAMPLES_REQUIRED)

                # Trạng thái faces
                face_txt = f"Faces: {len(detections)}"
                _put(display, face_txt, (10, 70), WHITE, 0.42)

                if quality_ok:
                    _put(display, "Quality: OK  [SPACE] de chup", (10, 90), GREEN, 0.42)
                elif best is not None:
                    _put(display, "Quality: FAIL", (10, 90), RED, 0.42)
                else:
                    _put(display, "Khong phat hien khuon mat", (10, 90), YELLOW, 0.42)

                # Thông điệp phản hồi (tồn tại 2.5 giây)
                if time.time() - last_msg_time < 2.5 and last_msg:
                    _put(display, last_msg, (10, cam_h - 40), last_msg_color, 0.5, 2)

                # Phím tắt góc dưới
                _put(display, "[SPC/S]=chup  [F]=finalize  [R]=reset  [N]=moi  [Q]=thoat",
                     (10, cam_h - 14), (160, 160, 160), 0.38)

                # FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                _put(display, f"FPS:{fps:.1f}", (cam_w - 70, 20), (160, 160, 160), 0.42)

                cv2.imshow("FaceAttend - Stage 7: Enrollment", display)
                key = cv2.waitKey(1) & 0xFF

                # --- Xử lý phím ---
                if key in (ord("q"), 27):   # Q hoặc ESC
                    break

                elif key in (ord("s"), ord(" ")):   # Chụp mẫu
                    result = service.add_sample(current_emp_id, frame)
                    if result.accepted:
                        last_msg       = f"[OK] {result.message}"
                        last_msg_color = GREEN
                    elif result.status == SampleStatus.MAX_REACHED:
                        last_msg       = f"[MAX] {result.message}"
                        last_msg_color = ORANGE
                    else:
                        last_msg       = f"[FAIL] {result.message}"
                        last_msg_color = RED
                    last_msg_time = time.time()
                    print(f"  [{result.status.value.upper()}] {result.message}")

                elif key == ord("f"):   # Finalize
                    final = service.finalize(current_emp_id)
                    if final.success:
                        # Lưu gallery vector ra file npz
                        gallery.save(GALLERY_FILE_PATH)
                        last_msg       = f"[THANH CONG] {final.message}"
                        last_msg_color = GREEN
                        print(f"\n  ✓ {final.message}")
                        print(f"  [Lưu vector] Đã lưu gallery vào: {GALLERY_FILE_PATH.relative_to(ROOT)}")
                    else:
                        last_msg       = f"[CHUA DU MAU] {final.message}"
                        last_msg_color = YELLOW
                        print(f"\n  ! {final.message}")
                    last_msg_time = time.time()

                elif key == ord("r"):   # Reset pending
                    service.reset_pending(current_emp_id)
                    last_msg       = "Da xoa toan bo mau chua finalize."
                    last_msg_color = YELLOW
                    last_msg_time  = time.time()
                    print("  [R] Đã reset pending.")

                elif key == ord("n"):   # Nhân viên mới
                    cv2.destroyAllWindows()
                    new_id, new_name, new_dept = _input_employee_info()
                    try:
                        service.create_profile(new_id, new_name, department=new_dept)
                        current_emp_id = new_id
                        print(f"\n  Hồ sơ: {new_id} — {new_name}. Tiếp tục chụp mẫu ...")
                    except ValueError as e:
                        print(f"  [LỖI] {e}")
                    last_msg = ""

    except RuntimeError as exc:
        print(f"\n[LỖI CAMERA] {exc}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()

    # --- Tổng kết ---
    if gallery.size > 0:
        gallery.save(GALLERY_FILE_PATH)

    total_time = time.time() - start_time
    avg_fps    = frame_count / total_time if total_time > 0 else 0
    summary    = service.summary()
    print("\n" + "=" * 50)
    print("  KẾT THÚC PHIÊN ĐĂNG KÝ")
    print("=" * 50)
    print(f"  Thời gian    : {total_time:.1f}s  |  FPS TB: {avg_fps:.1f}")
    print(f"  Nhân viên    : {summary['total']} (active: {summary['active']})")
    print(f"  Đã đăng ký   : {summary['enrolled']} người có đủ mẫu")
    print(f"  Gallery      : {summary['gallery_size']} người | {summary['gallery_samples']} mẫu tổng")
    print()
    for p in service.all_profiles:
        status_icon = "✓" if p.sample_count >= MIN_SAMPLES_REQUIRED else "·"
        print(f"  {status_icon}  {p}")


if __name__ == "__main__":
    run(camera_index=0, conf=0.5)

