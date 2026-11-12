"""
app/vision/face_quality.py
Kiểm tra chất lượng khuôn mặt đã căn chỉnh trước khi trích xuất embedding.

Các tiêu chí kiểm tra:
  1. Kích thước bbox tối thiểu (px) trên frame gốc
  2. Độ sắc nét (Laplacian variance) — phát hiện mờ
  3. Độ sáng trung bình — phát hiện quá tối / quá sáng
  4. Góc nghiêng đầu (yaw) — ước lượng từ vị trí 2 mắt và mũi
  5. Độ tự tin phát hiện (confidence) từ SCRFD

Tất cả tiêu chí đều có ngưỡng cấu hình được.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum lý do thất bại
# ---------------------------------------------------------------------------

class QualityFailReason(Enum):
    """Lý do khuôn mặt không đạt chất lượng."""
    TOO_SMALL       = auto()   # bbox quá nhỏ trên frame gốc
    TOO_BLURRY      = auto()   # ảnh mờ (Laplacian variance thấp)
    TOO_DARK        = auto()   # thiếu sáng
    TOO_BRIGHT      = auto()   # quá sáng (overexposed)
    TOO_TILTED      = auto()   # khuôn mặt nghiêng quá nhiều (yaw)
    LOW_CONFIDENCE  = auto()   # confidence SCRFD thấp


# ---------------------------------------------------------------------------
# Kết quả kiểm tra chất lượng
# ---------------------------------------------------------------------------

@dataclass
class QualityResult:
    """
    Kết quả kiểm tra chất lượng một khuôn mặt.

    Thuộc tính:
        passed          : True nếu đạt tất cả tiêu chí
        fail_reasons    : danh sách lý do thất bại (rỗng nếu passed=True)
        sharpness       : Laplacian variance (cao = sắc nét)
        brightness      : giá trị mean kênh V trong HSV (0–255)
        yaw_deg         : góc nghiêng ngang ước lượng (độ; 0 = thẳng)
        face_size_px    : cạnh nhỏ hơn của bbox gốc (px)
        confidence      : confidence từ detector (0–1)
        message         : thông điệp hướng dẫn người dùng (tiếng Việt)
    """
    passed: bool
    fail_reasons: list[QualityFailReason] = field(default_factory=list)
    sharpness: float = 0.0
    brightness: float = 0.0
    yaw_deg: float = 0.0
    face_size_px: float = 0.0
    confidence: float = 0.0
    message: str = ""

    def __str__(self) -> str:
        status = "ĐẠT" if self.passed else "KHÔNG ĐẠT"
        parts = [f"[{status}]"]
        parts.append(f"sharpness={self.sharpness:.1f}")
        parts.append(f"brightness={self.brightness:.1f}")
        parts.append(f"yaw={self.yaw_deg:.1f}°")
        parts.append(f"size={self.face_size_px:.0f}px")
        parts.append(f"conf={self.confidence:.2f}")
        if self.fail_reasons:
            parts.append(f"fail={[r.name for r in self.fail_reasons]}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Ngưỡng mặc định
# ---------------------------------------------------------------------------

@dataclass
class QualityThresholds:
    """
    Ngưỡng kiểm tra chất lượng. Tất cả có thể ghi đè khi khởi tạo FaceQualityChecker.

    Thuộc tính:
        min_face_size_px   : cạnh bbox tối thiểu trên frame gốc (px)
        min_sharpness      : Laplacian variance tối thiểu (ảnh sắc nét ≈ 80–200+)
        min_brightness     : độ sáng V tối thiểu (0–255)
        max_brightness     : độ sáng V tối đa (0–255)
        max_yaw_deg        : góc yaw tối đa (độ, tính từ 0=thẳng)
        min_confidence     : confidence SCRFD tối thiểu
    """
    min_face_size_px: float = 60.0
    min_sharpness: float = 40.0
    min_brightness: float = 30.0
    max_brightness: float = 230.0
    max_yaw_deg: float = 35.0
    min_confidence: float = 0.45


# ---------------------------------------------------------------------------
# Hàm đo lường nội bộ
# ---------------------------------------------------------------------------

def _measure_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance: cao = sắc nét, thấp = mờ."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _measure_brightness(aligned_bgr: np.ndarray) -> float:
    """
    Độ sáng trung bình kênh V trong HSV.
    Trả về giá trị [0, 255].
    """
    hsv = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def _estimate_yaw(landmarks: np.ndarray) -> float:
    """
    Ước lượng góc nghiêng ngang (yaw) từ 5 landmarks.

    Landmarks thứ tự InsightFace:
      0 = mắt trái, 1 = mắt phải, 2 = mũi, 3 = miệng trái, 4 = miệng phải

    Dùng sự lệch ngang giữa 2 mắt so với tổng chiều rộng liên mắt để ước lượng.
    Khi thẳng: eye_mid_x ≈ nose_x.
    Khi nghiêng trái: eye_mid_x > nose_x (và ngược lại).

    Trả về góc yaw ước lượng (độ). 0 = thẳng. Dấu dương = nghiêng phải.
    """
    if landmarks is None or landmarks.shape != (5, 2):
        return 0.0

    left_eye  = landmarks[0]
    right_eye = landmarks[1]
    nose      = landmarks[2]

    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_dist  = abs(right_eye[0] - left_eye[0])

    if eye_dist < 1.0:
        return 0.0

    # offset ngang mũi so với trung điểm 2 mắt, chuẩn hóa theo khoảng liên mắt
    offset_ratio = (nose[0] - eye_mid_x) / eye_dist
    # Ước lượng thô: 1.0 ratio ~ 45° yaw (heuristic cho ảnh webcam thực tế)
    yaw_deg = float(np.degrees(np.arctan(offset_ratio * 1.5)))
    return yaw_deg


# ---------------------------------------------------------------------------
# Lớp kiểm tra chất lượng chính
# ---------------------------------------------------------------------------

class FaceQualityChecker:
    """
    Kiểm tra chất lượng khuôn mặt trước khi đưa vào ArcFace.

    Tham số:
        thresholds : QualityThresholds — ngưỡng kiểm tra (None = dùng mặc định)
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def check(
        self,
        aligned_face_bgr: np.ndarray,
        *,
        confidence: float = 1.0,
        face_bbox: Optional[tuple[float, float, float, float]] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> QualityResult:
        """
        Kiểm tra chất lượng ảnh khuôn mặt đã căn chỉnh.

        Tham số:
            aligned_face_bgr : ndarray (H, W, 3) BGR — ảnh đã align (112×112)
            confidence       : confidence từ SCRFD
            face_bbox        : (x1, y1, x2, y2) trên frame gốc — dùng đo kích thước
            landmarks        : ndarray (5, 2) trên frame gốc — dùng ước lượng yaw

        Trả về:
            QualityResult
        """
        t = self.thresholds
        fail_reasons: list[QualityFailReason] = []

        # --- 1. Kích thước bbox ---
        face_size = 0.0
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            face_size = min(x2 - x1, y2 - y1)
            if face_size < t.min_face_size_px:
                fail_reasons.append(QualityFailReason.TOO_SMALL)

        # --- 2. Confidence ---
        if confidence < t.min_confidence:
            fail_reasons.append(QualityFailReason.LOW_CONFIDENCE)

        # --- 3. Độ sắc nét (Laplacian variance trên ảnh xám) ---
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = _measure_sharpness(gray)
        if sharpness < t.min_sharpness:
            fail_reasons.append(QualityFailReason.TOO_BLURRY)

        # --- 4. Độ sáng ---
        brightness = _measure_brightness(aligned_face_bgr)
        if brightness < t.min_brightness:
            fail_reasons.append(QualityFailReason.TOO_DARK)
        elif brightness > t.max_brightness:
            fail_reasons.append(QualityFailReason.TOO_BRIGHT)

        # --- 5. Góc yaw ---
        yaw_deg = _estimate_yaw(landmarks) if landmarks is not None else 0.0
        if abs(yaw_deg) > t.max_yaw_deg:
            fail_reasons.append(QualityFailReason.TOO_TILTED)

        passed = len(fail_reasons) == 0

        # Tạo thông điệp hướng dẫn người dùng
        message = _build_message(fail_reasons)

        result = QualityResult(
            passed=passed,
            fail_reasons=fail_reasons,
            sharpness=sharpness,
            brightness=brightness,
            yaw_deg=yaw_deg,
            face_size_px=face_size,
            confidence=confidence,
            message=message,
        )
        logger.debug("QualityCheck: %s", result)
        return result


def _build_message(reasons: list[QualityFailReason]) -> str:
    """Tạo thông điệp tiếng Việt hướng dẫn người dùng điều chỉnh."""
    if not reasons:
        return "Khuôn mặt đạt chất lượng."

    hints = {
        QualityFailReason.TOO_SMALL:      "Vui lòng đứng gần camera hơn.",
        QualityFailReason.TOO_BLURRY:     "Ảnh bị mờ — đứng yên và nhìn thẳng vào camera.",
        QualityFailReason.TOO_DARK:       "Thiếu ánh sáng — di chuyển đến nơi sáng hơn.",
        QualityFailReason.TOO_BRIGHT:     "Ánh sáng quá mạnh — tránh ánh sáng ngược.",
        QualityFailReason.TOO_TILTED:     "Khuôn mặt nghiêng quá nhiều — nhìn thẳng vào camera.",
        QualityFailReason.LOW_CONFIDENCE: "Không phát hiện rõ khuôn mặt — điều chỉnh góc nhìn.",
    }
    return " ".join(hints[r] for r in reasons)

