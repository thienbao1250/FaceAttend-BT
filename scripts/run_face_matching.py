"""
scripts/run_face_matching.py
Demo Giai đoạn 6 — Face Matching (nhận diện danh tính) trên webcam live.

Pipeline đầy đủ:
  Camera → SCRFD detect → align → quality check → ArcFace embedding → FaceMatcher

Phím điều khiển:
  R   : Đăng ký (Register) người đang đứng trước camera với tên nhập từ terminal
  D   : Xóa (Delete) toàn bộ gallery
  Q   : Thoát

Hiển thị:
  - Bbox màu xanh lá = KNOWN | vàng = UNCERTAIN | đỏ = UNKNOWN / quality fail
  - Tên nhân viên + cosine similarity trực tiếp trên frame
  - Panel phải: aligned face + thông tin match

Chạy: python scripts/run_face_matching.py
"""

import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.camera.camera_capture import CameraCapture
from app.vision.face_alignment import align_face, crop_face_simple, ALIGNED_SIZE
from app.vision.face_detector import FaceDetector
from app.vision.face_matcher import FaceGallery, FaceMatcher, MatchStatus, MatchThresholds
from app.vision.face_quality import FaceQualityChecker, QualityThresholds
from app.vision.face_recognizer import FaceRecognizer

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DETECTOR_PATH   = ROOT / "models" / "det_10g.onnx"
RECOGNIZER_PATH = ROOT / "models" / "w600k_r50.onnx"
FONT         = cv2.FONT_HERSHEY_SIMPLEX
PREVIEW_SIZE = ALIGNED_SIZE * 2  # 224×224

# Màu
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
YELLOW = (0, 200, 220)
CYAN   = (220, 200, 0)
WHITE  = (220, 220, 220)
GRAY   = (140, 140, 140)


# ---------------------------------------------------------------------------
# Tiện ích vẽ
# ---------------------------------------------------------------------------

def _draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> None:
    colors = [(255, 100, 100), (100, 100, 255), (100, 255, 100),
              (255, 255, 100), (100, 255, 255)]
    h, w = frame.shape[:2]
    for i, (lx, ly) in enumerate(landmarks.astype(int)):
        lx, ly = int(np.clip(lx, 0, w - 1)), int(np.clip(ly, 0, h - 1))
        cv2.circle(frame, (lx, ly), 4, colors[i], -1)
        cv2.circle(frame, (lx, ly), 4, (255, 255, 255), 1)


def _status_color(status: MatchStatus | None) -> tuple:
    if status == MatchStatus.KNOWN:
        return GREEN
    if status == MatchStatus.UNCERTAIN:
        return YELLOW
    return RED


def _put_lines(img, lines, x, y0, dy=18, scale=0.42, thick=1):
    for i, (txt, clr) in enumerate(lines):
        cv2.putText(img, txt, (x, y0 + i * dy), FONT, scale, clr, thick)


# ---------------------------------------------------------------------------
# Thread nhập tên từ terminal (không block vòng lặp camera)
# ---------------------------------------------------------------------------

class _InputThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.pending_name: str | None = None
        self._want_input = threading.Event()
        self._done       = threading.Event()

    def request(self):
        """Yêu cầu nhập tên từ main thread."""
        self.pending_name = None
        self._done.clear()
        self._want_input.set()

    def wait_result(self, timeout: float = 15.0) -> str | None:
        self._done.wait(timeout)
        return self.pending_name

    def run(self):
        while True:
            self._want_input.wait()
            self._want_input.clear()
            name = input("\n[Register] Nhập tên / mã nhân viên (Enter để bỏ qua): ").strip()
            self.pending_name = name if name else None
            self._done.set()


# ---------------------------------------------------------------------------
# Vòng lặp chính
# ---------------------------------------------------------------------------

def run(camera_index: int = 0, conf: float = 0.5) -> None:
    for p in [DETECTOR_PATH, RECOGNIZER_PATH]:
        if not p.exists():
            print(f"[LỖI] Model không tìm thấy: {p}")
            sys.exit(1)

    print(f"Tải model detector : {DETECTOR_PATH.name} ...")
    detector = FaceDetector(DETECTOR_PATH, conf_threshold=conf, nms_threshold=0.3, min_face_size=30)

    print(f"Tải model recognizer: {RECOGNIZER_PATH.name} ...")
    recognizer = FaceRecognizer(RECOGNIZER_PATH)

    checker = FaceQualityChecker(QualityThresholds(
        min_face_size_px=50, min_sharpness=35.0,
        min_brightness=25.0, max_brightness=235.0,
        max_yaw_deg=35.0, min_confidence=0.45,
    ))

    gallery   = FaceGallery()
    thresholds = MatchThresholds(accept=0.35, reject=0.20, margin_min=0.07)
    matcher   = FaceMatcher(gallery, thresholds)

    input_thread = _InputThread()
    input_thread.start()

    _registering  = False   # đang chờ nhập tên
    _pending_emb: np.ndarray | None = None   # embedding cần đăng ký

    print(
        "\nSẵn sàng!\n"
        "  [R]  : Đăng ký mặt hiện tại (quality phải PASSED)\n"
        "  [D]  : Xóa toàn bộ gallery\n"
        "  [Q]  : Thoát\n"
        f"  Ngưỡng: accept={thresholds.accept} | reject={thresholds.reject} "
        f"| margin_min={thresholds.margin_min}\n"
    )

    blank_preview = np.zeros((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8)

    try:
        with CameraCapture(camera_index=camera_index) as cam:
            cam_info = cam.get_info()
            cam_h, cam_w = cam_info["height"], cam_info["width"]
            print(f"Camera: {cam_w}×{cam_h}")

            start      = time.time()
            frame_cnt  = 0

            while True:
                # --- Nếu đang chờ input và thread đã xong ---
                if _registering and input_thread._done.is_set():
                    name = input_thread.pending_name
                    _registering = False
                    if name and _pending_emb is not None:
                        gallery.add(name, _pending_emb)
                        print(f"  → Đã đăng ký '{name}' | "
                              f"gallery: {gallery.size} người, {gallery.total_samples} mẫu")
                    else:
                        print("  → Hủy đăng ký.")
                    _pending_emb = None

                frame = cam.read_frame()
                frame_cnt += 1

                detections = detector.detect(frame)
                best = FaceDetector.largest_face(detections)

                preview        = blank_preview.copy()
                match_result   = None
                current_emb    = None
                quality_result = None

                if best is not None:
                    # Alignment
                    if best.landmarks is not None:
                        aligned_obj = align_face(frame, best.landmarks)
                        aligned_img = aligned_obj.image
                    else:
                        aligned_img = crop_face_simple(frame, best.bbox)

                    # Quality
                    quality_result = checker.check(
                        aligned_img,
                        confidence=best.confidence,
                        face_bbox=best.bbox,
                        landmarks=best.landmarks,
                    )

                    # Embedding & matching (chỉ khi quality OK)
                    if quality_result.passed:
                        try:
                            current_emb  = recognizer.get_embedding(aligned_img)
                            match_result = matcher.match(current_emb)
                        except Exception as exc:
                            logging.warning("Pipeline lỗi: %s", exc)

                    # --- Vẽ lên frame gốc ---
                    x1, y1, x2, y2 = best.as_int_bbox()
                    h_f, w_f = frame.shape[:2]
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w_f - 1, x2), min(h_f - 1, y2)

                    if match_result is not None:
                        box_clr = _status_color(match_result.status)
                    elif quality_result and not quality_result.passed:
                        box_clr = RED
                    else:
                        box_clr = GRAY

                    if x2c > x1c and y2c > y1c:
                        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), box_clr, 2)
                    if best.landmarks is not None:
                        _draw_landmarks(frame, best.landmarks)

                    # Label danh tính
                    if match_result is not None:
                        if match_result.is_known:
                            label = f"{match_result.employee_id}  {match_result.similarity:.2f}"
                        elif match_result.is_uncertain:
                            label = f"UNCERTAIN  {match_result.similarity:.2f}"
                        else:
                            label = f"UNKNOWN  {match_result.similarity:.2f}"
                    elif quality_result and not quality_result.passed:
                        label = quality_result.message
                    else:
                        label = f"conf {best.confidence:.2f}"

                    cv2.putText(frame, label,
                                (x1c + 2, max(y1c - 8, 14)),
                                FONT, 0.6, box_clr, 2)

                    # --- Preview panel ---
                    preview = cv2.resize(aligned_img, (PREVIEW_SIZE, PREVIEW_SIZE),
                                         interpolation=cv2.INTER_NEAREST)
                    info_lines = []
                    if quality_result:
                        q_clr = GREEN if quality_result.passed else RED
                        info_lines.append((f"quality: {'OK' if quality_result.passed else 'FAIL'}", q_clr))
                        info_lines.append((f"sharp:{quality_result.sharpness:.0f} bright:{quality_result.brightness:.0f}", WHITE))
                        info_lines.append((f"yaw:{quality_result.yaw_deg:+.1f}deg conf:{quality_result.confidence:.2f}", WHITE))

                    if match_result is not None:
                        m_clr = _status_color(match_result.status)
                        info_lines.append((f"status: {match_result.status.value}", m_clr))
                        info_lines.append((f"sim:    {match_result.similarity:.4f}", m_clr))
                        if match_result.employee_id:
                            info_lines.append((f"id:     {match_result.employee_id}", GREEN))
                        if match_result.runner_up_id:
                            info_lines.append((f"2nd:    {match_result.runner_up_id}({match_result.runner_up_similarity:.3f})", GRAY))
                        info_lines.append((f"n_cand: {match_result.num_candidates}", WHITE))
                        info_lines.append((f"time:   {match_result.latency_ms:.1f}ms", WHITE))

                    _put_lines(preview, info_lines, x=4, y0=10, dy=18, scale=0.38)

                # --- Gallery HUD ở dưới frame ---
                hud2 = (f"Gallery: {gallery.size} employees / {gallery.total_samples} samples"
                        "  |  [R]=register  [D]=clear  [Q]=quit")
                cv2.putText(frame, hud2, (8, cam_h - 10), FONT, 0.42, CYAN, 1)

                # Ghép frame + preview
                if PREVIEW_SIZE < cam_h:
                    pad = cam_h - PREVIEW_SIZE
                    preview = cv2.copyMakeBorder(preview, 0, pad, 0, 0,
                                                  cv2.BORDER_CONSTANT, value=0)
                else:
                    preview = preview[:cam_h]

                canvas = np.hstack([cv2.resize(frame, (cam_w, cam_h)), preview])

                # FPS HUD
                elapsed = time.time() - start
                fps = frame_cnt / elapsed if elapsed > 0 else 0
                hud = f"FPS: {fps:.1f} | faces: {len(detections)}"
                cv2.putText(canvas, hud, (10, 24), FONT, 0.65, (255, 255, 0), 2)

                cv2.imshow("FaceAttend - Stage 6: Face Matching", canvas)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("r"):
                    if current_emb is not None and not _registering:
                        _pending_emb = current_emb.copy()
                        _registering = True
                        input_thread.request()
                        print("[R] Nhìn vào camera và nhập tên trong terminal...")
                    elif current_emb is None:
                        print("[R] Không có embedding hợp lệ — quality chưa đạt?")
                    else:
                        print("[R] Đang chờ nhập tên — hoàn tất trước khi đăng ký mới.")

                elif key == ord("d"):
                    gallery.clear()
                    print(f"[D] Đã xóa toàn bộ gallery.")

    except RuntimeError as exc:
        print(f"Lỗi camera: {exc}")
        sys.exit(1)

    total = time.time() - start
    avg_fps = frame_cnt / total if total > 0 else 0
    print(f"\nTổng kết: {frame_cnt} frames | {total:.1f}s | FPS trung bình: {avg_fps:.1f}")
    print(f"Gallery cuối: {gallery.size} employees / {gallery.total_samples} samples")


if __name__ == "__main__":
    run(camera_index=0, conf=0.5)

