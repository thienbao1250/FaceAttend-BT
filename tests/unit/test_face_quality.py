"""
tests/unit/test_face_quality.py
Unit test cho app/vision/face_quality.py
Chạy: python -m pytest tests/unit/test_face_quality.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.vision.face_quality import (
    FaceQualityChecker,
    QualityFailReason,
    QualityResult,
    QualityThresholds,
    _estimate_yaw,
    _measure_brightness,
    _measure_sharpness,
)
from app.vision.face_alignment import ARCFACE_DST


# ---------------------------------------------------------------------------
# Helpers tạo ảnh synthetic
# ---------------------------------------------------------------------------

def _uniform_face(h: int = 112, w: int = 112, value: int = 128) -> np.ndarray:
    """Ảnh BGR đồng nhất (mờ hoàn toàn, sharpness rất thấp)."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _sharp_face(h: int = 112, w: int = 112) -> np.ndarray:
    """Ảnh BGR có cạnh sắc nét (Laplacian variance cao)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Vẽ các đường kẻ trắng → nhiều cạnh → sharpness cao
    for i in range(0, h, 10):
        img[i, :] = 200
    for j in range(0, w, 10):
        img[:, j] = 200
    return img


def _dark_face(h: int = 112, w: int = 112) -> np.ndarray:
    return np.full((h, w, 3), 10, dtype=np.uint8)


def _bright_face(h: int = 112, w: int = 112) -> np.ndarray:
    return np.full((h, w, 3), 250, dtype=np.uint8)


def _frontal_landmarks() -> np.ndarray:
    """Landmarks gần chuẩn thẳng."""
    return ARCFACE_DST.copy()


def _tilted_landmarks(offset: float = 50.0) -> np.ndarray:
    """Landmarks với mũi lệch mạnh → yaw lớn."""
    lm = ARCFACE_DST.copy()
    lm[2, 0] += offset   # dịch mũi sang phải
    return lm


# ===========================================================================
# _measure_sharpness
# ===========================================================================

class TestMeasureSharpness:
    def test_uniform_image_low_sharpness(self):
        gray = np.full((112, 112), 128, dtype=np.uint8)
        s = _measure_sharpness(gray)
        assert s < 1.0   # ảnh đồng nhất → Laplacian ≈ 0

    def test_sharp_image_high_sharpness(self):
        img = _sharp_face()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = _measure_sharpness(gray)
        assert s > 100.0   # ảnh có cạnh sắc → variance cao

    def test_returns_float(self):
        gray = np.zeros((112, 112), dtype=np.uint8)
        assert isinstance(_measure_sharpness(gray), float)


# ===========================================================================
# _measure_brightness
# ===========================================================================

class TestMeasureBrightness:
    def test_dark_image(self):
        b = _measure_brightness(_dark_face())
        assert b < 30.0

    def test_bright_image(self):
        b = _measure_brightness(_bright_face())
        assert b > 220.0

    def test_mid_image(self):
        b = _measure_brightness(_uniform_face(value=128))
        assert 100.0 < b < 160.0

    def test_returns_float(self):
        assert isinstance(_measure_brightness(_uniform_face()), float)


# ===========================================================================
# _estimate_yaw
# ===========================================================================

class TestEstimateYaw:
    def test_frontal_low_yaw(self):
        lm = _frontal_landmarks()
        yaw = _estimate_yaw(lm)
        assert abs(yaw) < 5.0   # gần thẳng

    def test_tilted_high_yaw(self):
        lm = _tilted_landmarks(offset=40.0)
        yaw = _estimate_yaw(lm)
        assert abs(yaw) > 20.0

    def test_none_landmarks_returns_zero(self):
        yaw = _estimate_yaw(None)
        assert yaw == 0.0

    def test_wrong_shape_returns_zero(self):
        yaw = _estimate_yaw(np.zeros((3, 2)))
        assert yaw == 0.0

    def test_returns_float(self):
        assert isinstance(_estimate_yaw(_frontal_landmarks()), float)

    def test_right_tilt_positive_yaw(self):
        """Mũi dịch phải → yaw dương."""
        lm = _frontal_landmarks()
        lm[2, 0] += 30.0
        yaw = _estimate_yaw(lm)
        assert yaw > 0.0

    def test_left_tilt_negative_yaw(self):
        """Mũi dịch trái → yaw âm."""
        lm = _frontal_landmarks()
        lm[2, 0] -= 30.0
        yaw = _estimate_yaw(lm)
        assert yaw < 0.0


# ===========================================================================
# QualityResult
# ===========================================================================

class TestQualityResult:
    def test_str_passed(self):
        r = QualityResult(passed=True, message="ok")
        s = str(r)
        assert "ĐẠT" in s

    def test_str_failed(self):
        r = QualityResult(
            passed=False,
            fail_reasons=[QualityFailReason.TOO_BLURRY],
            message="Mờ",
        )
        s = str(r)
        assert "KHÔNG ĐẠT" in s


# ===========================================================================
# FaceQualityChecker
# ===========================================================================

class TestFaceQualityChecker:
    def _checker(self, **kwargs):
        t = QualityThresholds(**kwargs)
        return FaceQualityChecker(t)

    # --- passed khi mọi thứ ổn ---
    def test_sharp_normal_frontal_passes(self):
        checker = FaceQualityChecker(QualityThresholds(
            min_face_size_px=0,  # bỏ qua size (không truyền bbox)
            min_sharpness=50.0,
            min_brightness=20.0,
            max_brightness=240.0,
            max_yaw_deg=40.0,
            min_confidence=0.0,
        ))
        face = _sharp_face()
        result = checker.check(face, confidence=0.9, landmarks=_frontal_landmarks())
        assert result.passed

    # --- TOO_BLURRY ---
    def test_uniform_face_fails_blurry(self):
        checker = FaceQualityChecker(QualityThresholds(min_sharpness=50.0))
        face = _uniform_face()   # sharpness ≈ 0
        result = checker.check(face, confidence=0.9)
        assert QualityFailReason.TOO_BLURRY in result.fail_reasons
        assert not result.passed

    # --- TOO_DARK ---
    def test_dark_face_fails(self):
        checker = FaceQualityChecker(QualityThresholds(
            min_brightness=30.0,
            min_sharpness=0.0,   # bỏ qua sharpness
        ))
        result = checker.check(_dark_face(), confidence=0.9)
        assert QualityFailReason.TOO_DARK in result.fail_reasons

    # --- TOO_BRIGHT ---
    def test_bright_face_fails(self):
        checker = FaceQualityChecker(QualityThresholds(
            max_brightness=200.0,
            min_sharpness=0.0,
        ))
        result = checker.check(_bright_face(), confidence=0.9)
        assert QualityFailReason.TOO_BRIGHT in result.fail_reasons

    # --- TOO_TILTED ---
    def test_tilted_face_fails(self):
        checker = FaceQualityChecker(QualityThresholds(
            max_yaw_deg=20.0,
            min_sharpness=0.0,
        ))
        lm = _tilted_landmarks(offset=50.0)
        result = checker.check(_sharp_face(), confidence=0.9, landmarks=lm)
        assert QualityFailReason.TOO_TILTED in result.fail_reasons

    # --- LOW_CONFIDENCE ---
    def test_low_confidence_fails(self):
        checker = FaceQualityChecker(QualityThresholds(
            min_confidence=0.6,
            min_sharpness=0.0,
        ))
        result = checker.check(_sharp_face(), confidence=0.4)
        assert QualityFailReason.LOW_CONFIDENCE in result.fail_reasons

    # --- TOO_SMALL ---
    def test_small_bbox_fails(self):
        checker = FaceQualityChecker(QualityThresholds(
            min_face_size_px=80.0,
            min_sharpness=0.0,
        ))
        result = checker.check(
            _sharp_face(),
            confidence=0.9,
            face_bbox=(0.0, 0.0, 50.0, 50.0),   # 50×50 < 80px
        )
        assert QualityFailReason.TOO_SMALL in result.fail_reasons

    # --- Nhiều lý do fail cùng lúc ---
    def test_multiple_fail_reasons(self):
        checker = FaceQualityChecker(QualityThresholds(
            min_sharpness=50.0,
            min_brightness=50.0,
            min_confidence=0.6,
            min_face_size_px=0.0,
        ))
        # Ảnh tối + mờ (uniform dark) + confidence thấp
        result = checker.check(_dark_face(), confidence=0.3)
        assert len(result.fail_reasons) >= 2
        assert QualityFailReason.TOO_DARK in result.fail_reasons

    # --- Thông điệp không rỗng khi fail ---
    def test_message_not_empty_on_fail(self):
        checker = FaceQualityChecker(QualityThresholds(min_sharpness=500.0))
        result = checker.check(_uniform_face(), confidence=0.9)
        assert not result.passed
        assert len(result.message) > 0

    # --- Thông điệp hợp lý khi pass ---
    def test_message_on_pass(self):
        checker = FaceQualityChecker(QualityThresholds(
            min_sharpness=0.0,
            min_confidence=0.0,
        ))
        result = checker.check(_sharp_face(), confidence=0.9)
        assert result.passed
        assert "đạt" in result.message.lower() or len(result.message) > 0

    # --- sharpness, brightness được đo đúng ---
    def test_sharpness_measured(self):
        checker = FaceQualityChecker(QualityThresholds(min_sharpness=0.0))
        result = checker.check(_sharp_face(), confidence=0.9)
        assert result.sharpness > 0.0

    def test_brightness_measured(self):
        checker = FaceQualityChecker(QualityThresholds(min_sharpness=0.0))
        result = checker.check(_uniform_face(value=128), confidence=0.9)
        assert 100.0 < result.brightness < 160.0

    # --- confidence được lưu lại ---
    def test_confidence_stored(self):
        checker = FaceQualityChecker(QualityThresholds(min_sharpness=0.0))
        result = checker.check(_uniform_face(), confidence=0.77)
        assert abs(result.confidence - 0.77) < 1e-5

