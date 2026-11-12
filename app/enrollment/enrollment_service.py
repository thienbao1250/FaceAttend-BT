"""
app/enrollment/enrollment_service.py
Đăng ký khuôn mặt nhân viên — Giai đoạn 7.

Luồng đăng ký:
  1. Tạo hồ sơ nhân viên (employee_id + thông tin cơ bản).
  2. Thu nhiều ảnh khuôn mặt; mỗi ảnh qua: detect → align → quality → embed.
  3. Lọc mẫu lỗi / chất lượng thấp; kiểm tra độ nhất quán giữa các mẫu.
  4. Lưu embedding vào FaceGallery gắn với employee_id.
  5. Cho phép cập nhật thêm mẫu, vô hiệu hóa hoặc xóa hồ sơ.

Bảo mật / Quyền riêng tư:
  - Không lưu ảnh gốc theo mặc định; chỉ giữ embedding đã chuẩn hóa.
  - Embedding là dữ liệu sinh trắc học nhạy cảm: hạn chế log, không in raw.
  - Quyền xóa và cập nhật hồ sơ phải được kiểm soát ở tầng gọi bên trên.

Thiết kế:
  - Module này không phụ thuộc camera hay UI; chỉ nhận ndarray ảnh đầu vào.
  - Tích hợp dễ vào CLI, webcam loop, hay FastAPI endpoint về sau.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.vision.face_detector import FaceDetector, FaceDetection
from app.vision.face_alignment import align_face, crop_face_simple
from app.vision.face_quality import FaceQualityChecker, QualityResult, QualityThresholds
from app.vision.face_recognizer import FaceRecognizer
from app.vision.face_matcher import FaceGallery

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

# Số mẫu tối thiểu để hoàn tất đăng ký
MIN_SAMPLES_REQUIRED: int = 3

# Số mẫu tối đa lưu cho một nhân viên (tránh gallery phình quá lớn)
MAX_SAMPLES_PER_EMPLOYEE: int = 10

# Ngưỡng cosine similarity tối thiểu giữa các mẫu của cùng một người
# (mẫu có thể hơi khác do góc, ánh sáng; nhưng quá thấp → có thể mẫu lỗi)
MIN_INTRA_CONSISTENCY: float = 0.15


# ---------------------------------------------------------------------------
# Enum trạng thái hồ sơ nhân viên
# ---------------------------------------------------------------------------

class EmployeeStatus(Enum):
    ACTIVE   = "active"    # Đang hoạt động, có thể nhận diện
    INACTIVE = "inactive"  # Tạm vô hiệu hóa (không xóa dữ liệu)
    DELETED  = "deleted"   # Đã xóa; hồ sơ chỉ còn trong audit log


# ---------------------------------------------------------------------------
# Hồ sơ nhân viên (in-memory, chưa gắn database)
# ---------------------------------------------------------------------------

@dataclass
class EmployeeProfile:
    """
    Thông tin cơ bản và trạng thái của một nhân viên.

    Thuộc tính:
        employee_id   : mã nhân viên duy nhất (string, bắt buộc)
        name          : họ tên hiển thị
        department    : phòng ban (tùy chọn)
        status        : EmployeeStatus — active / inactive / deleted
        sample_count  : số mẫu embedding đã đăng ký thành công
        enrolled_at   : unix timestamp thời điểm hoàn tất đăng ký lần đầu
        updated_at    : unix timestamp lần cập nhật mẫu gần nhất
        notes         : ghi chú tự do (không lưu dữ liệu sinh trắc học)
    """
    employee_id:  str
    name:         str
    department:   str            = ""
    status:       EmployeeStatus = EmployeeStatus.ACTIVE
    sample_count: int            = 0
    enrolled_at:  Optional[float] = None
    updated_at:   Optional[float] = None
    notes:        str            = ""

    def is_active(self) -> bool:
        return self.status == EmployeeStatus.ACTIVE

    def __str__(self) -> str:
        return (
            f"[{self.status.value.upper()}] {self.employee_id} | {self.name} "
            f"| {self.sample_count} mẫu"
        )


# ---------------------------------------------------------------------------
# Kết quả xử lý từng frame ảnh đầu vào
# ---------------------------------------------------------------------------

class SampleStatus(Enum):
    OK            = "ok"             # Mẫu hợp lệ, đã thêm vào gallery
    NO_FACE       = "no_face"        # Không phát hiện khuôn mặt
    MULTI_FACE    = "multi_face"     # Nhiều khuôn mặt — không rõ ai đăng ký
    LOW_QUALITY   = "low_quality"    # Quality check không đạt
    EMBED_ERROR   = "embed_error"    # Lỗi khi tạo embedding
    INCONSISTENT  = "inconsistent"   # Mẫu không nhất quán với các mẫu trước
    MAX_REACHED   = "max_reached"    # Đã đủ số mẫu tối đa


@dataclass
class SampleResult:
    """Kết quả xử lý một ảnh đầu vào trong quá trình đăng ký."""
    status:         SampleStatus
    message:        str                    = ""
    quality_result: Optional[QualityResult] = None
    embedding:      Optional[np.ndarray]   = None   # None nếu không hợp lệ

    @property
    def accepted(self) -> bool:
        return self.status == SampleStatus.OK


# ---------------------------------------------------------------------------
# Kết quả phiên đăng ký
# ---------------------------------------------------------------------------

@dataclass
class EnrollmentResult:
    """
    Kết quả tổng thể một phiên đăng ký (nhiều ảnh).

    Thuộc tính:
        success         : True nếu đủ mẫu hợp lệ để hoàn tất đăng ký
        employee_id     : mã nhân viên
        accepted        : số mẫu được chấp nhận
        rejected        : số mẫu bị từ chối
        message         : mô tả kết quả / hướng dẫn
        sample_results  : chi tiết từng mẫu (không lưu embedding thô)
    """
    success:        bool
    employee_id:    str
    accepted:       int                  = 0
    rejected:       int                  = 0
    message:        str                  = ""
    sample_results: list[SampleResult]   = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dịch vụ đăng ký nhân viên
# ---------------------------------------------------------------------------

class EnrollmentService:
    """
    Điều phối toàn bộ luồng đăng ký khuôn mặt cho một nhân viên.

    Tham số khởi tạo:
        detector   : FaceDetector (SCRFD) — phát hiện khuôn mặt
        recognizer : FaceRecognizer (ArcFace) — tạo embedding
        gallery    : FaceGallery — kho lưu embedding
        quality_checker : FaceQualityChecker (tùy chọn — tạo mặc định nếu None)
        min_samples     : số mẫu tối thiểu để hoàn tất đăng ký
        max_samples     : số mẫu tối đa lưu cho mỗi nhân viên
        min_consistency : ngưỡng cosine similarity giữa các mẫu (kiểm tra nhất quán)

    Luồng sử dụng::

        service = EnrollmentService(detector, recognizer, gallery)
        service.create_profile("EMP001", "Nguyễn Văn A", department="IT")

        for frame in captured_frames:           # list các ndarray BGR
            result = service.add_sample("EMP001", frame)
            print(result.status, result.message)

        final = service.finalize("EMP001")
        if final.success:
            print(f"Đăng ký thành công: {final.accepted} mẫu")
        else:
            print(f"Chưa đủ mẫu: {final.message}")
    """

    def __init__(
        self,
        detector:        FaceDetector,
        recognizer:      FaceRecognizer,
        gallery:         FaceGallery,
        quality_checker: Optional[FaceQualityChecker] = None,
        min_samples:     int   = MIN_SAMPLES_REQUIRED,
        max_samples:     int   = MAX_SAMPLES_PER_EMPLOYEE,
        min_consistency: float = MIN_INTRA_CONSISTENCY,
        save_images_dir: Optional[str | Path] = None,
    ) -> None:
        self._detector        = detector
        self._recognizer      = recognizer
        self._gallery         = gallery
        self._quality_checker = quality_checker or FaceQualityChecker(QualityThresholds())
        self._min_samples     = min_samples
        self._max_samples     = max_samples
        self._min_consistency = min_consistency
        self._save_images_dir = Path(save_images_dir) if save_images_dir else None

        # {employee_id: EmployeeProfile}
        self._profiles: dict[str, EmployeeProfile] = {}

        # Buffer tạm: embedding chưa finalize {employee_id: list[ndarray]}
        self._pending: dict[str, list[np.ndarray]] = {}
        # Buffer tạm: ảnh căn chỉnh (aligned 112x112) {employee_id: list[ndarray]}
        self._pending_images: dict[str, list[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Quản lý hồ sơ
    # ------------------------------------------------------------------

    def create_profile(
        self,
        employee_id: str,
        name:        str,
        department:  str = "",
        notes:       str = "",
        overwrite:   bool = False,
    ) -> EmployeeProfile:
        """
        Tạo hồ sơ nhân viên mới.

        Tham số:
            employee_id : mã nhân viên duy nhất (không được rỗng)
            name        : họ tên hiển thị
            department  : phòng ban (tùy chọn)
            notes       : ghi chú (không để dữ liệu sinh trắc học)
            overwrite   : True → cho phép tạo lại dù đã tồn tại
                          (xóa embedding cũ trong pending, giữ gallery)

        Raise:
            ValueError nếu employee_id hoặc name rỗng, hoặc đã tồn tại mà overwrite=False.
        """
        employee_id = employee_id.strip()
        name        = name.strip()

        if not employee_id:
            raise ValueError("employee_id không được để trống.")
        if not name:
            raise ValueError("name không được để trống.")

        if employee_id in self._profiles and not overwrite:
            raise ValueError(
                f"Nhân viên '{employee_id}' đã tồn tại. "
                "Dùng overwrite=True để tạo lại hoặc gọi update_profile()."
            )

        profile = EmployeeProfile(
            employee_id=employee_id,
            name=name,
            department=department,
            notes=notes,
        )
        self._profiles[employee_id] = profile
        self._pending[employee_id]  = []
        self._pending_images[employee_id] = []
        logger.info("create_profile: %s (%s)", employee_id, name)
        return profile

    def get_profile(self, employee_id: str) -> Optional[EmployeeProfile]:
        """Trả hồ sơ nhân viên hoặc None nếu không tồn tại."""
        return self._profiles.get(employee_id)

    def update_profile(
        self,
        employee_id: str,
        name:        Optional[str] = None,
        department:  Optional[str] = None,
        notes:       Optional[str] = None,
    ) -> EmployeeProfile:
        """
        Cập nhật thông tin cơ bản hồ sơ (không ảnh hưởng embedding).

        Raise:
            KeyError nếu employee_id không tồn tại.
        """
        profile = self._require_profile(employee_id)
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("name không được để trống.")
            profile.name = name
        if department is not None:
            profile.department = department.strip()
        if notes is not None:
            profile.notes = notes
        profile.updated_at = time.time()
        logger.info("update_profile: %s", employee_id)
        return profile

    def set_status(self, employee_id: str, status: EmployeeStatus) -> EmployeeProfile:
        """
        Thay đổi trạng thái nhân viên.

        - INACTIVE: nhân viên không được nhận diện nhưng dữ liệu vẫn còn.
        - DELETED : đánh dấu xóa logic; embedding thực sự bị xóa khỏi gallery.

        Raise:
            KeyError nếu employee_id không tồn tại.
        """
        profile = self._require_profile(employee_id)
        old_status = profile.status
        profile.status     = status
        profile.updated_at = time.time()

        if status == EmployeeStatus.DELETED:
            self._gallery.remove(employee_id)
            self._pending.pop(employee_id, None)
            self._pending_images.pop(employee_id, None)
            logger.info(
                "set_status: %s → DELETED; đã xóa embedding khỏi gallery.", employee_id
            )
        else:
            logger.info("set_status: %s %s → %s", employee_id, old_status.value, status.value)

        return profile

    def delete_profile(self, employee_id: str) -> bool:
        """
        Xóa hoàn toàn hồ sơ và embedding (xóa vật lý khỏi bộ nhớ).

        Trả về True nếu tìm thấy và xóa, False nếu không tồn tại.

        Lưu ý: Hành động này không thể hoàn tác trong phiên hiện tại.
        Trong môi trường thực tế phải ghi audit log trước khi xóa.
        """
        if employee_id not in self._profiles:
            return False
        self._gallery.remove(employee_id)
        self._pending.pop(employee_id, None)
        self._pending_images.pop(employee_id, None)
        del self._profiles[employee_id]
        logger.info("delete_profile: đã xóa vật lý hồ sơ %s", employee_id)
        return True

    @property
    def all_profiles(self) -> list[EmployeeProfile]:
        """Danh sách tất cả hồ sơ."""
        return list(self._profiles.values())

    @property
    def active_profiles(self) -> list[EmployeeProfile]:
        """Danh sách hồ sơ đang active."""
        return [p for p in self._profiles.values() if p.is_active()]

    # ------------------------------------------------------------------
    # Thêm mẫu (từng frame)
    # ------------------------------------------------------------------

    def add_sample(
        self,
        employee_id: str,
        frame_bgr:   np.ndarray,
        require_single_face: bool = True,
    ) -> SampleResult:
        """
        Xử lý một ảnh đầu vào và thêm mẫu embedding vào buffer pending.

        Tham số:
            employee_id         : mã nhân viên
            frame_bgr           : ảnh BGR (ndarray uint8, bất kỳ kích thước)
            require_single_face : True → từ chối nếu ảnh có nhiều mặt

        Trả về:
            SampleResult với status và message hướng dẫn.

        Raise:
            KeyError nếu employee_id chưa được tạo profile.
        """
        profile = self._require_profile(employee_id)

        # Kiểm tra giới hạn mẫu
        current_count = (
            len(self._pending.get(employee_id, []))
            + profile.sample_count
        )
        if current_count >= self._max_samples:
            return SampleResult(
                status=SampleStatus.MAX_REACHED,
                message=(
                    f"Đã đạt tối đa {self._max_samples} mẫu. "
                    "Xóa mẫu cũ trước khi thêm mới."
                ),
            )

        # --- Phát hiện khuôn mặt ---
        try:
            detections: list[FaceDetection] = self._detector.detect(frame_bgr)
        except Exception as exc:
            logger.warning("add_sample: detect lỗi: %s", exc)
            return SampleResult(
                status=SampleStatus.NO_FACE,
                message=f"Lỗi phát hiện khuôn mặt: {exc}",
            )

        if len(detections) == 0:
            return SampleResult(
                status=SampleStatus.NO_FACE,
                message="Không phát hiện khuôn mặt. Hãy nhìn thẳng vào camera.",
            )

        if require_single_face and len(detections) > 1:
            return SampleResult(
                status=SampleStatus.MULTI_FACE,
                message=(
                    f"Phát hiện {len(detections)} khuôn mặt. "
                    "Chỉ nên có một người trong khung hình khi đăng ký."
                ),
            )

        # Lấy khuôn mặt lớn nhất (gần camera nhất)
        best: FaceDetection = FaceDetector.largest_face(detections)  # type: ignore[assignment]

        # --- Căn chỉnh ---
        if best.landmarks is not None:
            aligned_result = align_face(frame_bgr, best.landmarks)
            aligned_img = aligned_result.image
        else:
            aligned_img = crop_face_simple(frame_bgr, best.bbox)

        # --- Kiểm tra chất lượng ---
        quality_result: QualityResult = self._quality_checker.check(
            aligned_img,
            confidence=best.confidence,
            face_bbox=best.bbox,
            landmarks=best.landmarks,
        )

        if not quality_result.passed:
            return SampleResult(
                status=SampleStatus.LOW_QUALITY,
                message=quality_result.message,
                quality_result=quality_result,
            )

        # --- Tạo embedding ---
        try:
            embedding = self._recognizer.get_embedding(aligned_img)
        except Exception as exc:
            logger.warning("add_sample: get_embedding lỗi: %s", exc)
            return SampleResult(
                status=SampleStatus.EMBED_ERROR,
                message=f"Lỗi tạo embedding: {exc}",
                quality_result=quality_result,
            )

        # --- Kiểm tra nhất quán với các mẫu đã có trong pending ---
        pending_embs = self._pending.get(employee_id, [])
        if pending_embs:
            consistency_ok, min_sim = self._check_consistency(embedding, pending_embs)
            if not consistency_ok:
                logger.warning(
                    "add_sample: mẫu không nhất quán (min_sim=%.3f < %.3f) — %s",
                    min_sim, self._min_consistency, employee_id,
                )
                return SampleResult(
                    status=SampleStatus.INCONSISTENT,
                    message=(
                        f"Mẫu không nhất quán với các mẫu trước "
                        f"(sim={min_sim:.3f} < {self._min_consistency:.2f}). "
                        "Có thể ảnh sai người hoặc điều kiện ánh sáng quá khác biệt."
                    ),
                    quality_result=quality_result,
                )

        # --- Chấp nhận mẫu ---
        if employee_id not in self._pending:
            self._pending[employee_id] = []
        self._pending[employee_id].append(embedding)

        if employee_id not in self._pending_images:
            self._pending_images[employee_id] = []
        self._pending_images[employee_id].append(aligned_img.copy())

        accepted_so_far = len(self._pending[employee_id])
        remaining = max(0, self._min_samples - accepted_so_far)
        msg = (
            f"Mẫu {accepted_so_far} đã lưu."
            + (f" Cần thêm {remaining} mẫu nữa." if remaining > 0 else " Đủ mẫu tối thiểu.")
        )

        logger.debug("add_sample: OK — %s mẫu %d", employee_id, accepted_so_far)
        return SampleResult(
            status=SampleStatus.OK,
            message=msg,
            quality_result=quality_result,
            embedding=None,   # Không trả embedding ra ngoài để hạn chế rò rỉ
        )

    # ------------------------------------------------------------------
    # Hoàn tất đăng ký
    # ------------------------------------------------------------------

    def finalize(self, employee_id: str) -> EnrollmentResult:
        """
        Hoàn tất phiên đăng ký: chuyển mẫu từ pending → gallery.

        - Yêu cầu tối thiểu min_samples mẫu hợp lệ.
        - Sau khi finalize, pending buffer của nhân viên được xóa.
        - Có thể gọi finalize nhiều lần để thêm mẫu (append).

        Tham số:
            employee_id : mã nhân viên

        Trả về:
            EnrollmentResult với success=True nếu đủ mẫu.

        Raise:
            KeyError nếu employee_id không tồn tại.
        """
        profile  = self._require_profile(employee_id)
        pending  = self._pending.get(employee_id, [])
        accepted = len(pending)
        now      = time.time()

        if accepted < self._min_samples:
            rejected = 0  # (thực tế cộng dồn từ add_sample; ở đây tính đơn giản)
            return EnrollmentResult(
                success=False,
                employee_id=employee_id,
                accepted=accepted,
                rejected=rejected,
                message=(
                    f"Chưa đủ mẫu: {accepted}/{self._min_samples}. "
                    "Tiếp tục thu thêm ảnh."
                ),
            )

        # Thêm vào gallery
        for emb in pending:
            self._gallery.add(employee_id, emb)

        # Lưu ảnh căn chỉnh ra đĩa nếu được cấu hình
        saved_images_count = 0
        if self._save_images_dir is not None:
            save_dir = self._save_images_dir / employee_id
            save_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(self._pending_images.get(employee_id, [])):
                file_path = save_dir / f"sample_{int(now)}_{i + 1:02d}.jpg"
                cv2.imwrite(str(file_path), img)
                saved_images_count += 1
            logger.info("finalize: Đã lưu %d ảnh khuôn mặt vào %s", saved_images_count, save_dir)

        # Cập nhật hồ sơ
        profile.sample_count += accepted
        profile.updated_at    = now
        if profile.enrolled_at is None:
            profile.enrolled_at = now

        # Xóa buffer pending
        self._pending[employee_id] = []
        self._pending_images[employee_id] = []

        logger.info(
            "finalize: %s — %d mẫu mới → tổng %d mẫu trong gallery",
            employee_id, accepted, profile.sample_count,
        )

        return EnrollmentResult(
            success=True,
            employee_id=employee_id,
            accepted=accepted,
            message=(
                f"Đăng ký thành công: {accepted} mẫu mới, "
                f"tổng {profile.sample_count} mẫu."
                + (f" Đã lưu {saved_images_count} ảnh vào {self._save_images_dir / employee_id}." if self._save_images_dir else "")
            ),
        )

    def pending_count(self, employee_id: str) -> int:
        """Số mẫu đang chờ finalize của nhân viên."""
        return len(self._pending.get(employee_id, []))

    def reset_pending(self, employee_id: str) -> None:
        """Hủy tất cả mẫu đang pending (chưa finalize) của nhân viên."""
        self._pending[employee_id] = []
        self._pending_images[employee_id] = []
        logger.info("reset_pending: đã xóa pending của %s", employee_id)

    # ------------------------------------------------------------------
    # Nội bộ
    # ------------------------------------------------------------------

    def _require_profile(self, employee_id: str) -> EmployeeProfile:
        """Trả profile hoặc raise KeyError nếu chưa tồn tại."""
        profile = self._profiles.get(employee_id)
        if profile is None:
            raise KeyError(
                f"Nhân viên '{employee_id}' chưa được tạo hồ sơ. "
                "Gọi create_profile() trước."
            )
        return profile

    def _check_consistency(
        self,
        new_emb:  np.ndarray,
        existing: list[np.ndarray],
    ) -> tuple[bool, float]:
        """
        Kiểm tra embedding mới có nhất quán với các mẫu hiện có không.

        Trả về:
            (passed: bool, min_similarity: float)
        """
        sims = [float(np.dot(new_emb, e)) for e in existing]
        min_sim = min(sims)
        return (min_sim >= self._min_consistency), min_sim

    # ------------------------------------------------------------------
    # Tiện ích thống kê
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Thống kê tổng hợp trạng thái đăng ký."""
        total    = len(self._profiles)
        active   = sum(1 for p in self._profiles.values() if p.status == EmployeeStatus.ACTIVE)
        inactive = sum(1 for p in self._profiles.values() if p.status == EmployeeStatus.INACTIVE)
        deleted  = sum(1 for p in self._profiles.values() if p.status == EmployeeStatus.DELETED)
        enrolled = sum(1 for p in self._profiles.values() if p.sample_count >= self._min_samples)
        return {
            "total":          total,
            "active":         active,
            "inactive":       inactive,
            "deleted":        deleted,
            "enrolled":       enrolled,
            "gallery_size":   self._gallery.size,
            "gallery_samples": self._gallery.total_samples,
        }

