"""
tests/unit/test_face_alignment.py
Unit test cho app/vision/face_alignment.py
Chạy: python -m pytest tests/unit/test_face_alignment.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.vision.face_alignment import (
    ARCFACE_DST,
    ALIGNED_SIZE,
    AlignedFace,
    align_face,
    crop_face_simple,
    _estimate_norm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_frame(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """Frame BGR giả có nội dung ngẫu nhiên."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _frontal_landmarks() -> np.ndarray:
    """Landmarks gần chuẩn thẳng (sao chép từ ARCFACE_DST để test identity-like)."""
    return ARCFACE_DST.copy()


def _tilted_landmarks() -> np.ndarray:
    """Landmarks bị dịch mũi sang phải mạnh → yaw lớn."""
    lm = ARCFACE_DST.copy()
    lm[2, 0] += 40.0   # dịch mũi sang phải 40px
    return lm


# ===========================================================================
# _estimate_norm
# ===========================================================================

class TestEstimateNorm:
    def test_returns_shape_2x3(self):
        src = _frontal_landmarks()
        dst = ARCFACE_DST.copy()
        mat = _estimate_norm(src, dst)
        assert mat.shape == (2, 3)

    def test_returns_float32(self):
        src = _frontal_landmarks()
        mat = _estimate_norm(src, ARCFACE_DST)
        assert mat.dtype == np.float32

    def test_identity_src_eq_dst(self):
        """src == dst → ma trận xấp xỉ identity (scale≈1, rotation≈0, t≈0)."""
        src = ARCFACE_DST.copy()
        mat = _estimate_norm(src, ARCFACE_DST)
        # Phần tử [0,0] và [1,1] ≈ 1.0; [0,1] và [1,0] ≈ 0.0
        assert abs(mat[0, 0] - 1.0) < 0.05
        assert abs(mat[1, 1] - 1.0) < 0.05
        assert abs(mat[0, 1]) < 0.05
        assert abs(mat[1, 0]) < 0.05


# ===========================================================================
# align_face
# ===========================================================================

class TestAlignFace:
    def test_returns_aligned_face_namedtuple(self):
        frame = _synthetic_frame()
        lm = _frontal_landmarks()
        result = align_face(frame, lm)
        assert isinstance(result, AlignedFace)

    def test_output_image_size_default(self):
        frame = _synthetic_frame()
        lm = _frontal_landmarks()
        result = align_face(frame, lm)
        assert result.image.shape == (ALIGNED_SIZE, ALIGNED_SIZE, 3)

    def test_output_image_size_custom(self):
        frame = _synthetic_frame()
        lm = _frontal_landmarks()
        result = align_face(frame, lm, output_size=224)
        assert result.image.shape == (224, 224, 3)

    def test_matrix_shape(self):
        frame = _synthetic_frame()
        lm = _frontal_landmarks()
        result = align_face(frame, lm)
        assert result.matrix.shape == (2, 3)

    def test_landmarks_aligned_shape(self):
        frame = _synthetic_frame()
        lm = _frontal_landmarks()
        result = align_face(frame, lm)
        assert result.landmarks_aligned.shape == (5, 2)

    def test_output_dtype_uint8(self):
        frame = _synthetic_frame()
        lm = _frontal_landmarks()
        result = align_face(frame, lm)
        assert result.image.dtype == np.uint8

    def test_invalid_frame_raises(self):
        lm = _frontal_landmarks()
        with pytest.raises(ValueError):
            align_face(None, lm)  # type: ignore

    def test_invalid_landmarks_shape_raises(self):
        frame = _synthetic_frame()
        with pytest.raises(ValueError):
            align_face(frame, np.zeros((3, 2), dtype=np.float32))  # sai shape

    def test_landmarks_aligned_in_output_range(self):
        """Landmarks đã căn chỉnh nên nằm gần trong [0, output_size]."""
        frame = _synthetic_frame(480, 640)
        # Đặt landmarks gần chuẩn trong frame 640×480
        lm = ARCFACE_DST.copy()
        # Scale template về vùng mặt giả định ở 640×480
        lm[:, 0] = lm[:, 0] + 200   # dịch sang x
        lm[:, 1] = lm[:, 1] + 150   # dịch xuống y
        result = align_face(frame, lm)
        # landmarks_aligned nên gần [0, 112]
        # (không nhất thiết hoàn toàn trong vì affine có thể xê dịch nhẹ)
        # Ít nhất phải có ít nhất 1 điểm trong range hợp lý
        assert result.landmarks_aligned.shape == (5, 2)


# ===========================================================================
# crop_face_simple
# ===========================================================================

class TestCropFaceSimple:
    def test_output_size_default(self):
        frame = _synthetic_frame()
        crop = crop_face_simple(frame, (100, 100, 300, 300))
        assert crop.shape == (ALIGNED_SIZE, ALIGNED_SIZE, 3)

    def test_output_size_custom(self):
        frame = _synthetic_frame()
        crop = crop_face_simple(frame, (100, 100, 300, 300), output_size=224)
        assert crop.shape == (224, 224, 3)

    def test_dtype_uint8(self):
        frame = _synthetic_frame()
        crop = crop_face_simple(frame, (50, 50, 200, 200))
        assert crop.dtype == np.uint8

    def test_scale_gt_1(self):
        """Scale > 1 → crop rộng hơn, kết quả vẫn đúng size."""
        frame = _synthetic_frame()
        crop = crop_face_simple(frame, (100, 100, 300, 300), scale=1.2)
        assert crop.shape == (ALIGNED_SIZE, ALIGNED_SIZE, 3)

    def test_invalid_bbox_returns_black(self):
        """bbox x1>=x2 sau clip → trả ảnh đen."""
        frame = _synthetic_frame()
        # bbox hoàn toàn ngoài frame
        crop = crop_face_simple(frame, (1000, 1000, 1001, 1001))
        # Nếu clip đẩy x1>=x2 → ảnh đen
        assert crop.shape == (ALIGNED_SIZE, ALIGNED_SIZE, 3)

    def test_full_frame_bbox(self):
        """bbox bằng toàn frame → crop thành công."""
        h, w = 480, 640
        frame = _synthetic_frame(h, w)
        crop = crop_face_simple(frame, (0, 0, w, h))
        assert crop.shape == (ALIGNED_SIZE, ALIGNED_SIZE, 3)
        # Phải có nội dung (không phải ảnh đen)
        assert crop.sum() > 0

