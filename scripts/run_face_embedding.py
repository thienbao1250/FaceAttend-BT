"""
scripts/run_face_embedding.py
Demo Giai đoạn 5 — Face Embedding (ArcFace) trên webcam live.

Hiển thị:
  - Frame gốc bên trái: bbox + landmarks từ SCRFD, trạng thái quality
  - Panel phải 224×224: aligned face + embedding info
  - Tính cosine similarity nếu nhấn S để lưu embedding "tham chiếu"

Phím điều khiển:
  S   : Lưu embedding hiện tại làm tham chiếu (người đăng ký mẫu)
  C   : Xóa embedding tham chiếu
  Q   : Thoát

Chạy: python scripts/run_face_embedding.py
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
from app.vision.face_recognizer import FaceRecognizer

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DETECTOR_PATH   = ROOT / "models" / "det_10g.onnx"
RECOGNIZER_PATH = ROOT / "models" / "w600k_r50.onnx"
FONT = cv2.FONT_HERSHEY_SIMPLEX
PREVIEW_SIZE = ALIGNED_SIZE * 2  # 224×224

# Màu sắc
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 220, 220)
CYAN   = (255, 220, 0)
WHITE  = (220, 220, 220)


# ---------------------------------------------------------------------------
# Tiện ích vẽ
# ---------------------------------------------------------------------------

def _draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> None:
    colors = [(255, 100, 100), (100, 100, 255), (100, 255, 100),
              (255, 255, 100), (100, 255, 255)]
    h, w = frame.shape[:2]
    for i, (lx, ly) in enumerate(landmarks.astype(int)):
        lx = int(np.clip(lx, 0, w - 1))
        ly = int(np.clip(ly, 0, h - 1))
        cv2.circle(frame, (lx, ly), 4, colors[i], -1)
        cv2.circle(frame, (lx, ly), 4, (255, 255, 255), 1)


def _draw_text_lines(
    img: np.ndarray,
    lines: list[tuple[str, tuple]],
    x: int,
    y_start: int,
    dy: int = 16,
    scale: float = 0.42,
    thickness: int = 1,
) -> None:
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (x, y_start + i * dy), FONT, scale, color, thickness)


def _similarity_color(sim: float) -> tuple:
    """Màu dựa trên ngưỡng similarity."""
    if sim >= 0.35:
        return GREEN
    if sim >= 0.20:
        return YELLOW
    return RED


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

    print(f"Tải model detector : {DETECTOR_PATH.name} ...")
    detector = FaceDetector(DETECTOR_PATH, conf_threshold=conf, nms_threshold=0.3, min_face_size=30)

    print(f"Tải model recognizer: {RECOGNIZER_PATH.name} ...")
    recognizer = FaceRecognizer(RECOGNIZER_PATH)

    checker = FaceQualityChecker(QualityThresholds(
        min_face_size_px=50, min_sharpness=35.0,
        min_brightness=25.0, max_brightness=235.0,
        max_yaw_deg=35.0, min_confidence=0.45,
    ))

    print(
        f"Sẵn sàng. | embedding_dim={recognizer.embedding_dim} | "
        "Nhấn [S] lưu tham chiếu | [C] xóa | [Q] thoát."
    )

    ref_embedding: np.ndarray | None = None   # embedding tham chiếu (người mẫu)
    blank_preview = np.zeros((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8)

    try:
        with CameraCapture(camera_index=camera_index) as cam:
            cam_info = cam.get_info()
            cam_h, cam_w = cam_info["height"], cam_info["width"]
            print(f"Camera: {cam_w}×{cam_h}")

            start = time.time()
            frame_count = 0

            while True:
                frame = cam.read_frame()
                frame_count += 1

                detections = detector.detect(frame)
                best = FaceDetector.largest_face(detections)

                preview = blank_preview.copy()
                current_emb: np.ndarray | None = None
                quality_result = None

                if best is not None:
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

                    # Embedding (chỉ khi quality đạt)
                    if quality_result.passed:
                        try:
                            current_emb = recognizer.get_embedding(aligned_img)
                        except Exception as exc:
                            logging.warning("get_embedding lỗi: %s", exc)

                    # Vẽ bbox lên frame
                    x1, y1, x2, y2 = best.as_int_bbox()
                    h_f, w_f = frame.shape[:2]
                    x1c = max(0, x1); y1c = max(0, y1)
                    x2c = min(w_f - 1, x2); y2c = min(h_f - 1, y2)
                    box_clr = GREEN if quality_result.passed else RED
                    if x2c > x1c and y2c > y1c:
                        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), box_clr, 2)
                    if best.landmarks is not None:
                        _draw_landmarks(frame, best.landmarks)

                    # Label confidence
                    cv2.putText(frame, f"{best.confidence:.2f}",
                                (x1c + 2, max(y1c - 6, 12)), FONT, 0.55, box_clr, 1)

                    # Thông điệp hướng dẫn
                    if not quality_result.passed:
                        guide_y = min(y2c + 18, h_f - 6)
                        cv2.putText(frame, quality_result.message, (x1c, guide_y),
                                    FONT, 0.42, RED, 1)

                    # Preview aligned
                    preview = cv2.resize(aligned_img, (PREVIEW_SIZE, PREVIEW_SIZE),
                                         interpolation=cv2.INTER_NEAREST)

                    # Overlay thông tin embedding
                    info_lines: list[tuple[str, tuple]] = [
                        (f"quality: {'OK' if quality_result.passed else 'FAIL'}", box_clr),
                        (f"sharp:   {quality_result.sharpness:.0f}", WHITE),
                        (f"bright:  {quality_result.brightness:.0f}", WHITE),
                        (f"yaw:     {quality_result.yaw_deg:+.1f}deg", WHITE),
                        (f"conf:    {quality_result.confidence:.2f}", WHITE),
                    ]

                    if current_emb is not None:
                        # Hiển thị vài thành phần đầu của embedding (debug)
                        snippet = " ".join(f"{v:+.2f}" for v in current_emb[:4])
                        info_lines.append((f"emb[:4]: {snippet}", CYAN))

                        if ref_embedding is not None:
                            sim = FaceRecognizer.cosine_similarity(current_emb, ref_embedding)
                            sim_color = _similarity_color(sim)
                            info_lines.append((f"sim ref: {sim:.4f}", sim_color))

                            # Thêm chú thích ngưỡng
                            if sim >= 0.35:
                                verdict = "MATCH (>=0.35)"
                            elif sim >= 0.20:
                                verdict = "UNCERTAIN"
                            else:
                                verdict = "NO MATCH (<0.20)"
                            info_lines.append((verdict, sim_color))
                    else:
                        if quality_result.passed:
                            info_lines.append(("emb: ERROR", RED))
                        else:
                            info_lines.append(("emb: skipped (quality)", YELLOW))

                    _draw_text_lines(preview, info_lines, x=4, y_start=10, dy=18, scale=0.38)

                # Chú thích tham chiếu
                if ref_embedding is not None:
                    cv2.putText(frame, "[REF saved] S=update C=clear", (8, cam_h - 10),
                                FONT, 0.45, GREEN, 1)
                else:
                    cv2.putText(frame, "[S] = save ref embedding", (8, cam_h - 10),
                                FONT, 0.45, YELLOW, 1)

                # Pad preview về chiều cao camera
                if PREVIEW_SIZE < cam_h:
                    pad = cam_h - PREVIEW_SIZE
                    preview = cv2.copyMakeBorder(preview, 0, pad, 0, 0,
                                                  cv2.BORDER_CONSTANT, value=0)
                else:
                    preview = preview[:cam_h]

                canvas = np.hstack([cv2.resize(frame, (cam_w, cam_h)), preview])

                # FPS HUD
                elapsed = time.time() - start
                fps = frame_count / elapsed if elapsed > 0 else 0
                hud = (f"FPS: {fps:.1f} | "
                       f"faces: {len(detections)} | "
                       f"model: {recognizer.model_name}")
                cv2.putText(canvas, hud, (10, 24), FONT, 0.65, (255, 255, 0), 2)

                cv2.imshow("FaceAttend - Stage 5: Face Embedding", canvas)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("s"):
                    if current_emb is not None:
                        ref_embedding = current_emb.copy()
                        print(f"[S] Đã lưu embedding tham chiếu (norm={np.linalg.norm(ref_embedding):.4f})")
                    else:
                        print("[S] Không lưu được: không có embedding hợp lệ (quality chưa đạt?)")
                elif key == ord("c"):
                    ref_embedding = None
                    print("[C] Đã xóa embedding tham chiếu.")

    except RuntimeError as exc:
        print(f"Lỗi camera: {exc}")
        sys.exit(1)

    total = time.time() - start
    avg_fps = frame_count / total if total > 0 else 0
    print(f"\nTổng kết: {frame_count} frames | {total:.1f}s | FPS trung bình: {avg_fps:.1f}")


if __name__ == "__main__":
    run(camera_index=0, conf=0.5)

