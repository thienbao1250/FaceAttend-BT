"""
app/vision/liveness.py
Phát hiện giả mạo khuôn mặt (Liveness Detection / Anti-Spoofing).

Mục tiêu: từ chối nhận diện khi người dùng đưa ảnh in hoặc màn hình video
           trước camera thay vì khuôn mặt thật 3D.

Phương pháp (passive, không cần model thêm):
  1. Texture Liveness (LBP variance)
       - Ảnh in / màn hình có cấu trúc texture đồng đều, phẳng, ít phương sai.
       - Khuôn mặt thật 3D có texture phong phú (lông mao, lỗ chân lông, bóng đổ).
       - Dùng Local Binary Pattern (LBP) trên vùng mũi–má để tính variance.

  2. Temporal Motion Check (multi-frame)
       - Yêu cầu khuôn mặt dịch chuyển nhẹ giữa các frame (micro-motion tự nhiên).
       - Ảnh tĩnh / video đứng yên sẽ bị từ chối sau một số frame liên tục không có
         chuyển động đủ.

  3. Specular Highlight Check
       - Màn hình điện thoại / máy tính thường phát ra ánh sáng đều và có vùng
         overexposed rất nhỏ.
       - Khuôn mặt thật có bóng sáng tự nhiên và ít điểm bão hòa.

Tích hợp:
  - Tạo LivenessChecker, gọi .check(face_bgr, face_bbox, landmarks) mỗi frame.
  - Trả LivenessResult.passed = True chỉ khi vượt tất cả bài kiểm tra.
  - Dùng LivenessTracker để tích lũy kết quả nhiều frame trước khi kết luận.

Lưu ý:
  - Độ chính xác passive liveness phụ thuộc điều kiện ánh sáng và chất lượng
    thiết bị in / màn hình. Nên kết hợp nhiều tín hiệu.
  - Các ngưỡng mặc định được hiệu chỉnh cho webcam thông thường ở điều kiện
    văn phòng; cần tinh chỉnh cho môi trường cụ thể.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum lý do thất bại
# ---------------------------------------------------------------------------

class LivenessFailReason(Enum):
    LOW_TEXTURE         = auto()   # Texture quá đồng đều → ảnh phẳng / in giấy
    NO_MOTION           = auto()   # Không đủ chuyển động trong N frame → ảnh tĩnh
    SCREEN_REFLECTION   = auto()   # Phát hiện phản chiếu màn hình (highlight bão hòa)
    INSUFFICIENT_FRAMES = auto()   # Chưa đủ frame để kết luận


# ---------------------------------------------------------------------------
# Kết quả kiểm tra liveness
# ---------------------------------------------------------------------------

@dataclass
class LivenessResult:
    """
    Kết quả kiểm tra liveness một frame.

    Thuộc tính:
        passed          : True chỉ khi tất cả bài kiểm tra vượt qua
        fail_reasons    : danh sách lý do thất bại
        texture_score   : LBP variance (cao = texture phong phú = thật)
        motion_score    : chỉ số chuyển động giữa frame này và frame trước
        specular_ratio  : tỉ lệ điểm bão hòa (thấp = thật)
        message         : thông điệp tiếng Việt
    """
    passed: bool
    fail_reasons: list[LivenessFailReason] = field(default_factory=list)
    texture_score: float = 0.0
    motion_score: float = 0.0
    specular_ratio: float = 0.0
    message: str = ""

    def __str__(self) -> str:
        status = "THẬT" if self.passed else "GIẢ MẠO"
        return (
            f"[{status}] texture={self.texture_score:.1f} "
            f"motion={self.motion_score:.4f} "
            f"specular={self.specular_ratio:.4f}"
            + (f" fail={[r.name for r in self.fail_reasons]}" if self.fail_reasons else "")
        )


# ---------------------------------------------------------------------------
# Ngưỡng mặc định
# ---------------------------------------------------------------------------

@dataclass
class LivenessThresholds:
    """
    Ngưỡng kiểm soát Liveness Detection.

    Thuộc tính:
        min_texture_score   : LBP std deviation tối thiểu để kết luận khuôn mặt thật.
                              Ảnh phẳng / đồng màu (giấy in) ≈ 0–30.
                              Ảnh màn hình điện thoại thường 30–55.
                              Khuôn mặt thật thường > 65 (nhiều texture tự nhiên).
                              Mặc định: 65.0
        min_motion_score    : Mean absolute difference tối thiểu giữa các frame (px).
                              Mặt thật có micro-motion tự nhiên > 0.4.
                              Ảnh tĩnh / video đứng yên ≈ 0. Mặc định: 0.4
        max_specular_ratio  : Tỉ lệ điểm bão hòa tối đa (pixel V > 250 / tổng pixel).
                              Màn hình sáng thường > 0.08. Mặc định: 0.07
        frames_required     : Số frame tích lũy tối thiểu để kết luận.
                              Mặc định: 8
        motion_frames_check : Bao nhiêu frame gần nhất cần có chuyển động.
                              Mặc định: 5
    """
    min_texture_score:   float = 65.0
    min_motion_score:    float = 0.4
    max_specular_ratio:  float = 0.07
    frames_required:     int   = 8
    motion_frames_check: int   = 5


# ---------------------------------------------------------------------------
# Hàm đo lường nội bộ
# ---------------------------------------------------------------------------

def _compute_lbp_variance(gray: np.ndarray) -> float:
    """
    Tính Local Binary Pattern (LBP) pixel variance trên ảnh xám.

    Thay vì histogram variance (không ổn định), dùng std deviation của
    các giá trị LBP pixel — bề mặt phẳng (ảnh in) có LBP giá trị đồng đều
    (std thấp), còn khuôn mặt thật 3D có LBP phong phú (std cao).

    Ảnh in / màn hình điện thoại:  std thường < 60
    Khuôn mặt thật:                std thường > 80
    """
    h, w = gray.shape
    gray_f = gray.astype(np.float32)

    # 8 hàng xóm theo thứ tự clockwise từ top-left
    neighbors = [
        gray_f[0:h-2, 0:w-2],  # top-left
        gray_f[0:h-2, 1:w-1],  # top
        gray_f[0:h-2, 2:w],    # top-right
        gray_f[1:h-1, 2:w],    # right
        gray_f[2:h,   2:w],    # bottom-right
        gray_f[2:h,   1:w-1],  # bottom
        gray_f[2:h,   0:w-2],  # bottom-left
        gray_f[1:h-1, 0:w-2],  # left
    ]
    center = gray_f[1:h-1, 1:w-1]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for i, nb in enumerate(neighbors):
        lbp += ((nb >= center).astype(np.uint8) << i)

    # Std deviation của giá trị LBP pixel (phong phú texture → std cao)
    return float(lbp.std())


def _compute_specular_ratio(face_bgr: np.ndarray) -> float:
    """
    Tính tỉ lệ điểm bão hòa (kênh V trong HSV > 250).
    Màn hình điện thoại/máy tính phát sáng mạnh → nhiều điểm bão hòa.
    Khuôn mặt thật ít điểm bão hòa hơn.
    """
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    saturated = np.sum(v_channel > 250)
    total = v_channel.size
    return float(saturated) / max(total, 1)


def _compute_motion(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Tính độ chuyển động giữa 2 frame xám bằng mean absolute difference.

    Dùng MAD thay vì optical flow để nhanh hơn và tránh phụ thuộc thêm thư viện.
    """
    if prev_gray.shape != curr_gray.shape:
        # Resize về cùng kích thước nếu khác nhau
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean())


def _crop_face_region(frame_bgr: np.ndarray, bbox: tuple) -> np.ndarray:
    """Cắt vùng khuôn mặt từ frame, với clamp để không vượt biên."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame_bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr
    return frame_bgr[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Lớp LivenessChecker — kiểm tra từng frame
# ---------------------------------------------------------------------------

class LivenessChecker:
    """
    Kiểm tra liveness một frame độc lập.

    Dùng LivenessTracker để tích lũy nhiều frame trước khi kết luận cuối cùng.
    """

    def __init__(self, thresholds: Optional[LivenessThresholds] = None) -> None:
        self.thresholds = thresholds or LivenessThresholds()

    def check_frame(
        self,
        face_bgr: np.ndarray,
        prev_gray: Optional[np.ndarray] = None,
    ) -> LivenessResult:
        """
        Kiểm tra liveness một frame.

        Tham số:
            face_bgr  : vùng khuôn mặt BGR (đã crop, kích thước bất kỳ)
            prev_gray : frame grayscale trước đó cùng vùng (để tính motion);
                        None nếu là frame đầu tiên.

        Trả về:
            LivenessResult
        """
        t = self.thresholds
        fail_reasons: list[LivenessFailReason] = []

        # Resize về 112x112 để chuẩn hóa phép đo
        face_resized = cv2.resize(face_bgr, (112, 112))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # --- 1. Texture LBP ---
        texture_score = _compute_lbp_variance(gray)
        if texture_score < t.min_texture_score:
            fail_reasons.append(LivenessFailReason.LOW_TEXTURE)

        # --- 2. Specular highlight ---
        specular_ratio = _compute_specular_ratio(face_resized)
        if specular_ratio > t.max_specular_ratio:
            fail_reasons.append(LivenessFailReason.SCREEN_REFLECTION)

        # --- 3. Motion (nếu có frame trước) ---
        motion_score = 0.0
        if prev_gray is not None:
            prev_resized = cv2.resize(prev_gray, (112, 112)) if prev_gray.shape[:2] != (112, 112) else prev_gray
            motion_score = _compute_motion(prev_resized, gray)
        # Motion check được xử lý ở cấp độ LivenessTracker (cần nhiều frame)

        passed = len(fail_reasons) == 0
        message = _build_liveness_message(fail_reasons)

        return LivenessResult(
            passed=passed,
            fail_reasons=fail_reasons,
            texture_score=texture_score,
            motion_score=motion_score,
            specular_ratio=specular_ratio,
            message=message,
        )


# ---------------------------------------------------------------------------
# Lớp LivenessTracker — tích lũy nhiều frame
# ---------------------------------------------------------------------------

class LivenessTracker:
    """
    Tích lũy kết quả nhiều frame để đưa ra quyết định liveness cuối cùng.

    Quy trình:
      1. Mỗi frame, gọi .update(face_bgr, bbox) → LivenessResult tức thời.
      2. Gọi .is_live() để lấy kết luận sau khi đủ frame.
      3. Gọi .reset() khi mất track khuôn mặt (không phát hiện được).

    Kết luận LIVE khi:
      - Đã đủ frames_required frame.
      - Tỉ lệ frame vượt texture & specular check >= texture_pass_rate.
      - Có đủ chuyển động trong motion_frames_check frame gần nhất.
    """

    def __init__(
        self,
        thresholds: Optional[LivenessThresholds] = None,
        texture_pass_rate: float = 0.6,
    ) -> None:
        self.thresholds       = thresholds or LivenessThresholds()
        self.texture_pass_rate = texture_pass_rate
        self._checker         = LivenessChecker(self.thresholds)

        # Lịch sử frame
        self._results:      deque[LivenessResult] = deque(maxlen=30)
        self._prev_gray:    Optional[np.ndarray]  = None
        self._motion_history: deque[float]        = deque(maxlen=20)

    def reset(self) -> None:
        """Reset khi mất track khuôn mặt hoặc bắt đầu phiên mới."""
        self._results.clear()
        self._motion_history.clear()
        self._prev_gray = None
        logger.debug("LivenessTracker: reset.")

    def update(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> LivenessResult:
        """
        Xử lý một frame mới.

        Tham số:
            frame_bgr : frame BGR gốc từ camera
            bbox      : (x1, y1, x2, y2) vùng khuôn mặt trên frame gốc

        Trả về:
            LivenessResult tức thời của frame này (chưa kết luận cuối cùng).
            Dùng .is_live() để lấy kết quả tích lũy.
        """
        face_crop = _crop_face_region(frame_bgr, bbox)
        result = self._checker.check_frame(face_crop, prev_gray=self._prev_gray)

        # Cập nhật frame trước
        face_resized = cv2.resize(face_crop, (112, 112))
        self._prev_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        self._results.append(result)
        self._motion_history.append(result.motion_score)

        logger.debug("LivenessTracker.update: %s", result)
        return result

    def is_live(self) -> tuple[bool, str]:
        """
        Kết luận liveness dựa trên tích lũy nhiều frame.

        Trả về:
            (is_live: bool, reason_message: str)
        """
        t = self.thresholds
        n = len(self._results)

        if n < t.frames_required:
            remaining = t.frames_required - n
            return False, f"⏳ Đang xác thực... ({remaining} frame còn lại)"

        recent = list(self._results)[-t.frames_required:]

        # --- Kiểm tra texture & specular trên các frame gần nhất ---
        texture_pass = sum(
            1 for r in recent
            if LivenessFailReason.LOW_TEXTURE not in r.fail_reasons
            and LivenessFailReason.SCREEN_REFLECTION not in r.fail_reasons
        )
        texture_rate = texture_pass / len(recent)

        if texture_rate < self.texture_pass_rate:
            avg_texture = sum(r.texture_score for r in recent) / len(recent)
            avg_specular = sum(r.specular_ratio for r in recent) / len(recent)
            if avg_specular > t.max_specular_ratio:
                return False, "🚫 Phát hiện màn hình điện thoại/máy tính — vui lòng sử dụng khuôn mặt thật."
            return False, "🚫 Phát hiện ảnh giả mạo (ảnh in/ảnh phẳng) — vui lòng đứng thẳng trước camera."

        # --- Kiểm tra chuyển động ---
        motion_window = list(self._motion_history)[-t.motion_frames_check:]
        if len(motion_window) >= t.motion_frames_check:
            avg_motion = sum(motion_window) / len(motion_window)
            if avg_motion < t.min_motion_score:
                return False, "🚫 Không phát hiện chuyển động tự nhiên — vui lòng cử động nhẹ."

        return True, "✅ Khuôn mặt thật xác thực thành công."

    @property
    def frame_count(self) -> int:
        return len(self._results)

    @property
    def latest_result(self) -> Optional[LivenessResult]:
        return self._results[-1] if self._results else None


# ---------------------------------------------------------------------------
# Hàm helper tạo thông điệp
# ---------------------------------------------------------------------------

def _build_liveness_message(reasons: list[LivenessFailReason]) -> str:
    if not reasons:
        return "✅ Frame đạt liveness."

    hints = {
        LivenessFailReason.LOW_TEXTURE:         "Ảnh phẳng / ảnh in được phát hiện.",
        LivenessFailReason.NO_MOTION:           "Không có chuyển động tự nhiên.",
        LivenessFailReason.SCREEN_REFLECTION:   "Phát hiện phản chiếu màn hình.",
        LivenessFailReason.INSUFFICIENT_FRAMES: "Chưa đủ frame để kết luận.",
    }
    return " | ".join(hints.get(r, r.name) for r in reasons)
