"""
app/vision/face_alignment.py
Căn chỉnh khuôn mặt (face alignment) dùng 5 facial landmarks từ SCRFD.

Pipeline:
  frame BGR + landmarks (5,2) → affine transform → crop 112×112 BGR

Chuẩn tọa độ template ArcFace/InsightFace (112×112):
  Mắt trái  : (38.29, 51.70)
  Mắt phải  : (73.53, 51.70)
  Mũi       : (56.02, 71.74)
  Miệng trái: (41.55, 92.37)
  Miệng phải: (70.73, 92.37)

Nguồn template: insightface / arcface_torch
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template 5 điểm chuẩn ArcFace 112×112
# ---------------------------------------------------------------------------

ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

ALIGNED_SIZE = 112  # kích thước output chuẩn ArcFace


# ---------------------------------------------------------------------------
# Kết quả alignment
# ---------------------------------------------------------------------------

class AlignedFace(NamedTuple):
    """
    Kết quả căn chỉnh một khuôn mặt.

    Thuộc tính:
        image       : ndarray (112, 112, 3) BGR — ảnh đã căn chỉnh
        matrix      : ndarray (2, 3) float32 — ma trận affine dùng để transform
        landmarks_aligned : ndarray (5, 2) — landmarks trong không gian 112×112
    """
    image: np.ndarray
    matrix: np.ndarray
    landmarks_aligned: np.ndarray


# ---------------------------------------------------------------------------
# Hàm nội bộ — ước lượng affine transform tối thiểu (5 điểm → 5 điểm)
# ---------------------------------------------------------------------------

def _estimate_norm(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Ước lượng ma trận affine 2×3 từ src (5,2) → dst (5,2).

    Dùng cv2.estimateAffinePartial2D (similarity transform: scale+rotate+translate)
    để giữ tỷ lệ khuôn mặt. Nếu không tính được, trả về ma trận đơn vị.

    Trả về: ndarray (2, 3) float32
    """
    # estimateAffinePartial2D cần shape (N,1,2) hoặc (N,2)
    mat, _ = cv2.estimateAffinePartial2D(
        src.reshape(-1, 1, 2),
        dst.reshape(-1, 1, 2),
        method=cv2.LMEDS,
    )
    if mat is None:
        logger.warning(
            "estimateAffinePartial2D thất bại — dùng ma trận đơn vị fallback."
        )
        mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return mat.astype(np.float32)


# ---------------------------------------------------------------------------
# API công khai
# ---------------------------------------------------------------------------

def align_face(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = ALIGNED_SIZE,
    dst_template: np.ndarray | None = None,
) -> AlignedFace:
    """
    Căn chỉnh khuôn mặt bằng affine transform từ 5 landmarks.

    Tham số:
        frame_bgr   : ảnh BGR gốc từ camera (H, W, 3)
        landmarks   : ndarray (5, 2) tọa độ 5 điểm mốc trong frame gốc
                      (thứ tự: mắt trái, mắt phải, mũi, miệng trái, miệng phải)
        output_size : kích thước ảnh output (mặc định 112)
        dst_template: template tọa độ đích; None → dùng ARCFACE_DST chuẩn

    Trả về:
        AlignedFace với ảnh 112×112 BGR, ma trận affine và landmarks đã transform.

    Raise:
        ValueError: nếu frame hoặc landmarks không hợp lệ.
    """
    if frame_bgr is None or frame_bgr.ndim != 3:
        raise ValueError("frame_bgr phải là ndarray 3 chiều (H, W, 3).")
    if landmarks is None or landmarks.shape != (5, 2):
        raise ValueError(
            f"landmarks phải có shape (5, 2), nhận được {getattr(landmarks, 'shape', None)}."
        )

    dst = dst_template if dst_template is not None else ARCFACE_DST

    # Scale template nếu output_size khác 112
    if output_size != ALIGNED_SIZE:
        scale = output_size / ALIGNED_SIZE
        dst = dst * scale

    src = landmarks.astype(np.float32)
    mat = _estimate_norm(src, dst)

    # Áp dụng affine transform
    aligned_img = cv2.warpAffine(
        frame_bgr,
        mat,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Tính landmarks trong không gian đã căn chỉnh (để debug/visualize)
    ones = np.ones((5, 1), dtype=np.float32)
    src_h = np.hstack([src, ones])           # (5, 3)
    lm_aligned = (mat @ src_h.T).T           # (5, 2)

    logger.debug(
        "align_face: mat=\n%s\noutput_size=%d", mat, output_size
    )
    return AlignedFace(image=aligned_img, matrix=mat, landmarks_aligned=lm_aligned)


def crop_face_simple(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    scale: float = 1.0,
    output_size: int = ALIGNED_SIZE,
) -> np.ndarray:
    """
    Cắt khuôn mặt theo bbox đơn giản (không dùng landmarks).

    Dùng khi model không có landmarks hoặc landmarks bị lỗi.

    Tham số:
        frame_bgr   : ảnh BGR gốc
        bbox        : (x1, y1, x2, y2) pixel
        scale       : nhân thêm vùng crop (1.0 = vừa bbox; 1.2 = thêm 20% bốn phía)
        output_size : kích thước resize output

    Trả về:
        ndarray (output_size, output_size, 3) BGR
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    h_frame, w_frame = frame_bgr.shape[:2]

    if scale != 1.0:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half_w = (x2 - x1) / 2 * scale
        half_h = (y2 - y1) / 2 * scale
        x1, y1 = cx - half_w, cy - half_h
        x2, y2 = cx + half_w, cy + half_h

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(w_frame - 1, x2))
    y2 = int(min(h_frame - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

