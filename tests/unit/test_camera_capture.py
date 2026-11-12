"""
tests/unit/test_camera_capture.py
Unit test cho CameraCapture — không cần camera thật (dùng mock).
Chạy: python -m pytest tests/unit/test_camera_capture.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.camera.camera_capture import CameraCapture


# ---------------------------------------------------------------------------
# Helper: tạo mock VideoCapture
# ---------------------------------------------------------------------------

def _make_mock_cap(opened=True, read_ok=True, w=640, h=480, fps=30.0):
    """Tạo mock cv2.VideoCapture với các thuộc tính cơ bản."""
    mock = MagicMock()
    mock.isOpened.return_value = opened

    dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
    mock.read.return_value = (read_ok, dummy_frame if read_ok else None)

    def _get(prop_id):
        mapping = {
            3: float(w),   # CAP_PROP_FRAME_WIDTH
            4: float(h),   # CAP_PROP_FRAME_HEIGHT
            5: fps,        # CAP_PROP_FPS
        }
        return mapping.get(prop_id, 0.0)

    mock.get.side_effect = _get
    return mock


# ---------------------------------------------------------------------------
# Kiểm tra mở camera thành công
# ---------------------------------------------------------------------------

@patch("app.camera.camera_capture.cv2.VideoCapture")
def test_open_success(mock_vc_cls):
    mock_cap = _make_mock_cap(opened=True)
    mock_vc_cls.return_value = mock_cap

    cam = CameraCapture(camera_index=0)
    cam.open()

    assert cam.is_opened is True
    info = cam.get_info()
    assert info["opened"] is True
    assert info["width"] == 640
    assert info["height"] == 480
    assert info["fps"] == 30.0

    cam.release()


# ---------------------------------------------------------------------------
# Kiểm tra mở camera thất bại (camera không khả dụng)
# ---------------------------------------------------------------------------

@patch("app.camera.camera_capture.cv2.VideoCapture")
def test_open_failure_raises(mock_vc_cls):
    mock_cap = _make_mock_cap(opened=False)
    mock_vc_cls.return_value = mock_cap

    cam = CameraCapture(camera_index=99)
    with pytest.raises(RuntimeError, match="Không thể mở camera"):
        cam.open()

    assert cam.is_opened is False


# ---------------------------------------------------------------------------
# Kiểm tra read_frame khi camera chưa mở
# ---------------------------------------------------------------------------

def test_read_frame_before_open():
    cam = CameraCapture()
    with pytest.raises(RuntimeError, match="Camera chưa được mở"):
        cam.read_frame()


# ---------------------------------------------------------------------------
# Kiểm tra read_frame thành công
# ---------------------------------------------------------------------------

@patch("app.camera.camera_capture.cv2.VideoCapture")
def test_read_frame_success(mock_vc_cls):
    mock_cap = _make_mock_cap(opened=True, read_ok=True, w=640, h=480)
    mock_vc_cls.return_value = mock_cap

    cam = CameraCapture()
    cam.open()
    frame = cam.read_frame()

    assert frame is not None
    assert frame.shape == (480, 640, 3)

    cam.release()


# ---------------------------------------------------------------------------
# Kiểm tra read_frame khi camera trả frame rỗng (đứt kết nối giữa chừng)
# ---------------------------------------------------------------------------

@patch("app.camera.camera_capture.cv2.VideoCapture")
def test_read_frame_failure_raises(mock_vc_cls):
    mock_cap = _make_mock_cap(opened=True, read_ok=False)
    mock_vc_cls.return_value = mock_cap

    cam = CameraCapture()
    cam.open()

    with pytest.raises(RuntimeError, match="Không đọc được frame"):
        cam.read_frame()

    cam.release()


# ---------------------------------------------------------------------------
# Kiểm tra context manager (__enter__ / __exit__)
# ---------------------------------------------------------------------------

@patch("app.camera.camera_capture.cv2.VideoCapture")
def test_context_manager(mock_vc_cls):
    mock_cap = _make_mock_cap(opened=True)
    mock_vc_cls.return_value = mock_cap

    with CameraCapture() as cam:
        assert cam.is_opened is True

    # Sau khi thoát context, camera phải đã được release
    mock_cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# Kiểm tra get_info khi camera chưa mở
# ---------------------------------------------------------------------------

def test_get_info_not_opened():
    cam = CameraCapture()
    info = cam.get_info()
    assert info == {"opened": False}


# ---------------------------------------------------------------------------
# Kiểm tra release an toàn khi gọi nhiều lần
# ---------------------------------------------------------------------------

@patch("app.camera.camera_capture.cv2.VideoCapture")
def test_release_idempotent(mock_vc_cls):
    mock_cap = _make_mock_cap(opened=True)
    mock_vc_cls.return_value = mock_cap

    cam = CameraCapture()
    cam.open()
    cam.release()
    cam.release()  # Lần 2 không được raise lỗi

    assert cam.is_opened is False

