"""
app/vision/preprocessing.py
Tiền xử lý frame cho pipeline nhận diện khuôn mặt.

Hai hàm chính:
- preprocess_for_detector(frame, input_size) → tensor SCRFD
- preprocess_for_recognizer(face_bgr)         → tensor ArcFace

Tất cả model trong dự án này đều dùng:
  - Màu: RGB (OpenCV mặc định BGR → phải chuyển đổi)
  - Normalize: (pixel - 127.5) / 128.0  → [-1.0, 1.0]
  - Format: NCHW float32 (batch=1, channel, height, width)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số chuẩn hoá (giống nhau cho cả SCRFD và ArcFace của insightface)
# ---------------------------------------------------------------------------
_NORM_MEAN = 127.5
_NORM_STD = 128.0

# Kích thước đầu vào mặc định của từng model
SCRFD_INPUT_SIZE = (640, 640)   # (width, height)
ARCFACE_INPUT_SIZE = (112, 112)  # (width, height)


# ---------------------------------------------------------------------------
# Dataclass chứa thông tin letterbox — dùng để ánh xạ tọa độ ngược lại
# ---------------------------------------------------------------------------
@dataclass
class LetterboxInfo:
    """
    Lưu tham số letterbox để có thể ánh xạ tọa độ từ ảnh đã scale
    về tọa độ gốc.

    Thuộc tính:
        scale       : tỷ lệ scale đồng nhất áp dụng cho cả chiều
        pad_left    : số pixel padding bên trái
        pad_top     : số pixel padding phía trên
        orig_w      : chiều rộng frame gốc
        orig_h      : chiều cao frame gốc
        input_w     : chiều rộng tensor đầu vào model
        input_h     : chiều cao tensor đầu vào model
    """
    scale: float
    pad_left: int
    pad_top: int
    orig_w: int
    orig_h: int
    input_w: int
    input_h: int

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        """Chuyển tọa độ trong không gian tensor về không gian frame gốc."""
        ox = (x - self.pad_left) / self.scale
        oy = (y - self.pad_top) / self.scale
        return ox, oy

    def scale_boxes_to_original(self, boxes: np.ndarray) -> np.ndarray:
        """
        Chuyển mảng bounding box (N, 4) dạng [x1, y1, x2, y2]
        từ không gian tensor về không gian frame gốc.
        """
        result = boxes.copy().astype(np.float32)
        result[:, [0, 2]] = (result[:, [0, 2]] - self.pad_left) / self.scale
        result[:, [1, 3]] = (result[:, [1, 3]] - self.pad_top) / self.scale
        return result

    def scale_landmarks_to_original(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Chuyển mảng landmarks (N, K, 2) hoặc (N, K*2)
        từ không gian tensor về không gian frame gốc.
        """
        result = landmarks.copy().astype(np.float32)
        if result.ndim == 3:
            result[:, :, 0] = (result[:, :, 0] - self.pad_left) / self.scale
            result[:, :, 1] = (result[:, :, 1] - self.pad_top) / self.scale
        else:
            # shape (N, K*2): x ở vị trí chẵn, y ở vị trí lẻ
            result[:, 0::2] = (result[:, 0::2] - self.pad_left) / self.scale
            result[:, 1::2] = (result[:, 1::2] - self.pad_top) / self.scale
        return result


# ---------------------------------------------------------------------------
# Hàm nội bộ: letterbox resize giữ tỷ lệ
# ---------------------------------------------------------------------------
def _letterbox(
    img_rgb: np.ndarray,
    target_w: int,
    target_h: int,
    pad_value: int = 0,
) -> tuple[np.ndarray, LetterboxInfo]:
    """
    Resize ảnh RGB giữ tỷ lệ và padding về kích thước target.

    Trả về:
        img_padded  : ảnh đã resize + padding, shape (target_h, target_w, 3)
        info        : LetterboxInfo chứa tham số để ánh xạ ngược tọa độ
    """
    orig_h, orig_w = img_rgb.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tính padding để căn giữa
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    img_padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    img_padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    info = LetterboxInfo(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        orig_w=orig_w,
        orig_h=orig_h,
        input_w=target_w,
        input_h=target_h,
    )
    return img_padded, info


# ---------------------------------------------------------------------------
# Hàm nội bộ: normalize + chuyển sang tensor NCHW
# ---------------------------------------------------------------------------
def _to_nchw_tensor(img_rgb: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh RGB (H, W, 3) uint8 thành tensor NCHW float32.
    Normalize: (pixel - 127.5) / 128.0 → [-1.0, 1.0]

    Trả về:
        ndarray shape (1, 3, H, W) dtype float32
    """
    blob = img_rgb.astype(np.float32)
    blob = (blob - _NORM_MEAN) / _NORM_STD       # normalize
    blob = blob.transpose(2, 0, 1)               # HWC → CHW
    blob = np.expand_dims(blob, axis=0)          # CHW → NCHW
    return blob


# ---------------------------------------------------------------------------
# API công khai
# ---------------------------------------------------------------------------

def preprocess_for_detector(
    frame_bgr: np.ndarray,
    input_size: tuple[int, int] = SCRFD_INPUT_SIZE,
) -> tuple[np.ndarray, LetterboxInfo]:
    """
    Tiền xử lý frame BGR từ OpenCV cho SCRFD detector.

    Các bước:
        1. BGR → RGB
        2. Letterbox resize giữ tỷ lệ về input_size, padding = 0
        3. Normalize (pixel - 127.5) / 128.0
        4. Chuyển thành tensor NCHW float32

    Tham số:
        frame_bgr  : ndarray (H, W, 3) uint8, BGR từ OpenCV
        input_size : (width, height) đầu vào model, mặc định (640, 640)

    Trả về:
        tensor    : ndarray (1, 3, H, W) float32
        lb_info   : LetterboxInfo — dùng để ánh xạ tọa độ bounding box về gốc
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("frame_bgr rỗng hoặc None.")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"frame_bgr phải có shape (H, W, 3), nhận được {frame_bgr.shape}"
        )

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    target_w, target_h = input_size
    padded, lb_info = _letterbox(img_rgb, target_w, target_h, pad_value=0)
    tensor = _to_nchw_tensor(padded)

    logger.debug(
        "preprocess_for_detector: %s → tensor %s, scale=%.4f, pad=(%d,%d)",
        frame_bgr.shape,
        tensor.shape,
        lb_info.scale,
        lb_info.pad_left,
        lb_info.pad_top,
    )
    return tensor, lb_info


def preprocess_for_recognizer(
    face_crop_bgr: np.ndarray,
    input_size: tuple[int, int] = ARCFACE_INPUT_SIZE,
) -> np.ndarray:
    """
    Tiền xử lý crop khuôn mặt BGR cho ArcFace recognizer.

    Các bước:
        1. BGR → RGB
        2. Resize trực tiếp về input_size (đã căn chỉnh ở bước alignment)
        3. Normalize (pixel - 127.5) / 128.0
        4. Chuyển thành tensor NCHW float32

    Tham số:
        face_crop_bgr : ndarray (H, W, 3) uint8, BGR — khuôn mặt đã crop+align
        input_size    : (width, height), mặc định (112, 112)

    Trả về:
        tensor : ndarray (1, 3, H, W) float32
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        raise ValueError("face_crop_bgr rỗng hoặc None.")
    if face_crop_bgr.ndim != 3 or face_crop_bgr.shape[2] != 3:
        raise ValueError(
            f"face_crop_bgr phải có shape (H, W, 3), nhận được {face_crop_bgr.shape}"
        )

    img_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    target_w, target_h = input_size
    resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    tensor = _to_nchw_tensor(resized)

    logger.debug(
        "preprocess_for_recognizer: %s → tensor %s",
        face_crop_bgr.shape,
        tensor.shape,
    )
    return tensor

