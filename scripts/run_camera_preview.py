"""
scripts/run_camera_preview.py
Chạy preview camera để kiểm tra module CameraCapture.
Nhấn Q để thoát.
"""

import logging
import sys
import time

import cv2

# Thêm root vào sys.path để import app
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from app.camera.camera_capture import CameraCapture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_preview(camera_index: int = 0, duration_sec: float | None = None) -> None:
    """
    Mở camera và hiển thị preview.

    Tham số:
        camera_index: Chỉ số camera (0 = mặc định).
        duration_sec: Thời gian chạy tối đa (giây). None = chạy đến khi nhấn Q.
    """
    try:
        with CameraCapture(camera_index=camera_index) as cam:
            info = cam.get_info()
            logger.info("Thông tin camera: %s", info)
            print(f"Camera info: {info}")
            print("Nhấn Q trong cửa sổ video để thoát.")

            start = time.time()
            frame_count = 0

            while True:
                frame = cam.read_frame()
                frame_count += 1

                # Hiển thị FPS thực tế lên frame
                elapsed = time.time() - start
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(
                    frame,
                    f"FPS: {fps_actual:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("FaceAttend AI - Camera Preview", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Người dùng nhấn Q, thoát.")
                    break

                if duration_sec is not None and elapsed >= duration_sec:
                    logger.info("Đã chạy đủ %.1f giây, thoát.", duration_sec)
                    break

            logger.info(
                "Tổng frame: %d | Thời gian: %.2fs | FPS trung bình: %.1f",
                frame_count,
                elapsed,
                frame_count / elapsed if elapsed > 0 else 0,
            )

    except RuntimeError as exc:
        logger.error("Lỗi camera: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    run_preview(camera_index=0)

