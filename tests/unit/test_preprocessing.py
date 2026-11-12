"""
tests/unit/test_preprocessing.py
Unit test cho app/vision/preprocessing.py
Chạy: python -m pytest tests/unit/test_preprocessing.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.vision.preprocessing import (
    ARCFACE_INPUT_SIZE,
    SCRFD_INPUT_SIZE,
    LetterboxInfo,
    preprocess_for_detector,
    preprocess_for_recognizer,
    _letterbox,
    _to_nchw_tensor,
)


# ---------------------------------------------------------------------------
# Helper: tạo frame giả
# ---------------------------------------------------------------------------

def _make_bgr(h: int, w: int) -> np.ndarray:
    """Frame BGR uint8 với giá trị ngẫu nhiên."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ===========================================================================
# _to_nchw_tensor
# ===========================================================================

class TestToNchwTensor:
    def test_shape_nchw(self):
        img = _make_bgr(100, 200)   # note: RGB truyền vào hàm này
        tensor = _to_nchw_tensor(img)
        assert tensor.shape == (1, 3, 100, 200), "Shape phải là NCHW"

    def test_dtype_float32(self):
        img = _make_bgr(50, 50)
        tensor = _to_nchw_tensor(img)
        assert tensor.dtype == np.float32

    def test_value_range(self):
        """Giá trị phải nằm trong [-1.0, 1.0] sau normalize."""
        img = _make_bgr(64, 64)
        tensor = _to_nchw_tensor(img)
        assert tensor.min() >= -1.0 - 1e-5
        assert tensor.max() <= 1.0 + 1e-5

    def test_pixel_0_maps_to_minus1(self):
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        tensor = _to_nchw_tensor(img)
        expected = (0 - 127.5) / 128.0
        assert abs(tensor[0, 0, 0, 0] - expected) < 1e-5

    def test_pixel_255_maps_to_near_plus1(self):
        img = np.full((1, 1, 3), 255, dtype=np.uint8)
        tensor = _to_nchw_tensor(img)
        expected = (255 - 127.5) / 128.0
        assert abs(tensor[0, 0, 0, 0] - expected) < 1e-5


# ===========================================================================
# _letterbox
# ===========================================================================

class TestLetterbox:
    def test_output_shape(self):
        img = _make_bgr(480, 640)  # treated as RGB
        padded, _ = _letterbox(img, 640, 640)
        assert padded.shape == (640, 640, 3)

    def test_output_dtype(self):
        img = _make_bgr(100, 200)
        padded, _ = _letterbox(img, 320, 320)
        assert padded.dtype == np.uint8

    def test_scale_landscape(self):
        """Frame ngang 640x480 → target 640x640: scale = min(1.0, 1.333) = 1.0"""
        img = _make_bgr(480, 640)
        _, info = _letterbox(img, 640, 640)
        expected_scale = min(640 / 640, 640 / 480)
        assert abs(info.scale - expected_scale) < 1e-4

    def test_scale_portrait(self):
        """Frame dọc 1280x720 → target 640x640: scale = min(0.5, 0.888) = 0.5"""
        img = _make_bgr(1280, 720)
        _, info = _letterbox(img, 640, 640)
        expected_scale = min(640 / 720, 640 / 1280)
        assert abs(info.scale - expected_scale) < 1e-4

    def test_padding_symmetry_square_input(self):
        """Ảnh vuông → padding trái/phải = padding trên/dưới = 0."""
        img = _make_bgr(300, 300)
        _, info = _letterbox(img, 640, 640)
        assert info.pad_left == 0
        assert info.pad_top == 0

    def test_padding_landscape_has_top_pad(self):
        """Frame ngang → scale giữ chiều rộng → padding phía trên/dưới > 0."""
        img = _make_bgr(360, 640)
        _, info = _letterbox(img, 640, 640)
        assert info.pad_top > 0

    def test_to_original_no_pad(self):
        """Không có pad → ánh xạ ngược chỉ chia scale."""
        img = _make_bgr(640, 640)
        _, info = _letterbox(img, 640, 640)
        # scale=1.0, pad=0 → tọa độ không đổi
        x, y = info.to_original(100.0, 200.0)
        assert abs(x - 100.0) < 1e-3
        assert abs(y - 200.0) < 1e-3

    def test_to_original_round_trip(self):
        """Điểm bất kỳ → letterbox → ánh xạ ngược phải về gần điểm gốc."""
        orig_h, orig_w = 480, 640
        img = _make_bgr(orig_h, orig_w)
        _, info = _letterbox(img, 640, 640)
        # Điểm (320, 240) trong ảnh gốc → trong tensor = (320*scale+pad_left, 240*scale+pad_top)
        tx = 320 * info.scale + info.pad_left
        ty = 240 * info.scale + info.pad_top
        rx, ry = info.to_original(tx, ty)
        assert abs(rx - 320) < 1.0
        assert abs(ry - 240) < 1.0

    def test_scale_boxes_to_original(self):
        img = _make_bgr(480, 640)
        _, info = _letterbox(img, 640, 640)
        # box trong tensor space
        boxes_tensor = np.array([[info.pad_left, info.pad_top,
                                  info.pad_left + 100 * info.scale,
                                  info.pad_top + 100 * info.scale]])
        boxes_orig = info.scale_boxes_to_original(boxes_tensor)
        assert abs(boxes_orig[0, 0]) < 1.0   # x1 ≈ 0
        assert abs(boxes_orig[0, 1]) < 1.0   # y1 ≈ 0
        assert abs(boxes_orig[0, 2] - 100) < 1.0
        assert abs(boxes_orig[0, 3] - 100) < 1.0


# ===========================================================================
# preprocess_for_detector
# ===========================================================================

class TestPreprocessForDetector:
    def test_output_tensor_shape(self):
        frame = _make_bgr(480, 640)
        tensor, _ = preprocess_for_detector(frame)
        assert tensor.shape == (1, 3, 640, 640)

    def test_output_dtype_float32(self):
        frame = _make_bgr(480, 640)
        tensor, _ = preprocess_for_detector(frame)
        assert tensor.dtype == np.float32

    def test_value_range(self):
        frame = _make_bgr(480, 640)
        tensor, _ = preprocess_for_detector(frame)
        assert tensor.min() >= -1.0 - 1e-5
        assert tensor.max() <= 1.0 + 1e-5

    def test_returns_letterbox_info(self):
        frame = _make_bgr(480, 640)
        _, info = preprocess_for_detector(frame)
        assert isinstance(info, LetterboxInfo)

    def test_no_nan_or_inf(self):
        frame = _make_bgr(480, 640)
        tensor, _ = preprocess_for_detector(frame)
        assert not np.isnan(tensor).any(), "Tensor không được chứa NaN"
        assert not np.isinf(tensor).any(), "Tensor không được chứa Inf"

    def test_custom_input_size(self):
        frame = _make_bgr(480, 640)
        tensor, _ = preprocess_for_detector(frame, input_size=(320, 320))
        assert tensor.shape == (1, 3, 320, 320)

    def test_empty_frame_raises(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            preprocess_for_detector(empty)

    def test_wrong_channels_raises(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises((ValueError, Exception)):
            preprocess_for_detector(gray)

    def test_none_raises(self):
        with pytest.raises((ValueError, AttributeError, Exception)):
            preprocess_for_detector(None)

    def test_bgr_rgb_conversion(self):
        """Pixel đỏ trong BGR phải trở thành đỏ trong RGB sau xử lý."""
        # Tạo frame hoàn toàn đỏ: BGR(0, 0, 255) = RGB(255, 0, 0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 2] = 255   # kênh R trong BGR
        tensor, _ = preprocess_for_detector(frame, input_size=(100, 100))
        # Sau chuyển BGR→RGB, channel 0 (R) phải ≈ (255-127.5)/128 ≈ 1.0
        # channel 1 (G) và channel 2 (B) phải ≈ (0-127.5)/128 ≈ -1.0
        assert tensor[0, 0, 0, 0] > 0.9    # channel R cao
        assert tensor[0, 1, 0, 0] < -0.9   # channel G thấp
        assert tensor[0, 2, 0, 0] < -0.9   # channel B thấp


# ===========================================================================
# preprocess_for_recognizer
# ===========================================================================

class TestPreprocessForRecognizer:
    def test_output_tensor_shape(self):
        face = _make_bgr(100, 100)
        tensor = preprocess_for_recognizer(face)
        assert tensor.shape == (1, 3, 112, 112)

    def test_output_dtype_float32(self):
        face = _make_bgr(100, 100)
        tensor = preprocess_for_recognizer(face)
        assert tensor.dtype == np.float32

    def test_value_range(self):
        face = _make_bgr(112, 112)
        tensor = preprocess_for_recognizer(face)
        assert tensor.min() >= -1.0 - 1e-5
        assert tensor.max() <= 1.0 + 1e-5

    def test_no_nan_or_inf(self):
        face = _make_bgr(112, 112)
        tensor = preprocess_for_recognizer(face)
        assert not np.isnan(tensor).any()
        assert not np.isinf(tensor).any()

    def test_various_input_sizes(self):
        """Crop bất kỳ kích thước → luôn ra 112x112."""
        for h, w in [(50, 50), (200, 150), (112, 112)]:
            face = _make_bgr(h, w)
            tensor = preprocess_for_recognizer(face)
            assert tensor.shape == (1, 3, 112, 112), f"Failed for input ({h},{w})"

    def test_custom_input_size(self):
        face = _make_bgr(100, 100)
        tensor = preprocess_for_recognizer(face, input_size=(64, 64))
        assert tensor.shape == (1, 3, 64, 64)

    def test_empty_face_raises(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            preprocess_for_recognizer(empty)

    def test_none_raises(self):
        with pytest.raises((ValueError, AttributeError, Exception)):
            preprocess_for_recognizer(None)

