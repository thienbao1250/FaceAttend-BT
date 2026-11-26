"""
app/database/models.py
SQLAlchemy ORM models — Giai đoạn 10.

Ánh xạ trực tiếp từ ERD đã thiết kế:
  - employees              : thông tin nhân viên
  - face_templates         : embedding khuôn mặt (binary bytea/BLOB)
  - cameras                : thông tin camera
  - face_recognition_logs  : log kỹ thuật mỗi lần camera nhận diện
  - attendance_events      : sự kiện check-in / check-out (legacy / realtime)
  - attendance_records     : bản ghi chính thức chấm công theo ngày
  - users                  : tài khoản quản trị
  - roles                  : vai trò và quyền
  - audit_logs             : nhật ký thao tác

Lưu ý bảo mật:
  - Cột `embedding` lưu dạng LargeBinary (bytea/BLOB) — không log giá trị này.
  - Cột `password_hash` trong users — không log giá trị này.
  - audit_logs.old_value / new_value KHÔNG được chứa embedding thô.

Chính sách lưu trữ (theo đề xuất nghiệp vụ):
  - Log nhận diện thành công  : giữ 30–90 ngày
  - Log nhận diện thất bại    : giữ 7–30 ngày
  - Ảnh chụp sự kiện          : giữ 7–30 ngày nếu cần
  - Bản ghi chấm công chính   : theo quy định pháp luật
  - Face Embedding            : trong thời gian còn mục đích nhận diện
"""

from __future__ import annotations

from datetime import date as date_type, datetime, time as time_type, timezone
from typing import Optional

import numpy as np
from sqlalchemy import (
    BigInteger, Boolean, Date, DateTime, Float, ForeignKey,
    Index, Integer, LargeBinary, String, Text, Time, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


# ---------------------------------------------------------------------------
# Helper: chuyển đổi embedding numpy ↔ bytes
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 512  # ArcFace w600k_r50 output dimension


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """
    Chuyển numpy array (512,) float32 sang bytes để lưu vào DB.
    Dùng struct pack để đảm bảo định dạng nhất quán.
    """
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Embedding phải có shape ({EMBEDDING_DIM},), nhận {embedding.shape}")
    return embedding.tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    """Khôi phục numpy array (512,) float32 từ bytes đã lưu trong DB."""
    return np.frombuffer(data, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# Hàm helper lấy timestamp UTC hiện tại
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Model: roles
# ---------------------------------------------------------------------------

class Role(Base):
    """
    Vai trò quản trị hệ thống.
    Ví dụ: admin, operator, viewer.
    """
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    permissions: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="JSON danh sách quyền, ví dụ: [\"employee:read\", \"attendance:write\"]"
    )

    # Relationship
    users: Mapped[list["User"]] = relationship("User", back_populates="role")

    def __repr__(self) -> str:
        return f"<Role id={self.id} name={self.name!r}>"


# ---------------------------------------------------------------------------
# Model: users
# ---------------------------------------------------------------------------

class User(Base):
    """Tài khoản quản trị hệ thống (không phải nhân viên chấm công)."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(
        String(255), nullable=False,
        comment="bcrypt hash — không log giá trị này"
    )
    role_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("roles.id", ondelete="SET NULL"), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    role: Mapped[Optional[Role]] = relationship("Role", back_populates="users")
    audit_logs: Mapped[list["AuditLog"]] = relationship("AuditLog", back_populates="user")

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} active={self.is_active}>"


# ---------------------------------------------------------------------------
# Model: employees
# ---------------------------------------------------------------------------

class Employee(Base):
    """
    Thông tin nhân viên.
    Dùng soft-delete (status='deleted') thay vì xóa hẳn để giữ lịch sử chấm công.
    """
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_code: Mapped[str] = mapped_column(
        String(50), nullable=False, unique=True,
        comment="Mã nhân viên nghiệp vụ (VD: NV001) — duy nhất"
    )
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, unique=True)
    department: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    position: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="active",
        comment="active | inactive | deleted"
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    face_templates: Mapped[list["FaceTemplate"]] = relationship(
        "FaceTemplate", back_populates="employee", cascade="all, delete-orphan"
    )
    attendance_events: Mapped[list["AttendanceEvent"]] = relationship(
        "AttendanceEvent", back_populates="employee"
    )
    face_recognition_logs: Mapped[list["FaceRecognitionLog"]] = relationship(
        "FaceRecognitionLog", back_populates="employee"
    )
    attendance_records: Mapped[list["AttendanceRecord"]] = relationship(
        "AttendanceRecord", back_populates="employee"
    )

    # Index
    __table_args__ = (
        Index("ix_employees_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Employee id={self.id} code={self.employee_code!r} name={self.full_name!r} status={self.status}>"


# ---------------------------------------------------------------------------
# Model: face_templates
# ---------------------------------------------------------------------------

class FaceTemplate(Base):
    """
    Embedding khuôn mặt của một nhân viên.

    Lưu dạng binary (LargeBinary / bytea) — dùng embedding_to_bytes() và
    bytes_to_embedding() để chuyển đổi.

    Một nhân viên có thể có nhiều FaceTemplate.
    Khi so khớp, chỉ dùng những template có is_active=True.

    QUAN TRỌNG: Cột `embedding` là dữ liệu sinh trắc học nhạy cảm.
    Không log, không serialize ra response API mà không có lý do nghiệp vụ rõ ràng.
    """
    __tablename__ = "face_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False
    )
    embedding: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False,
        comment="float32[512] dạng binary — KHÔNG LOG"
    )
    model_name: Mapped[str] = mapped_column(
        String(100), nullable=False,
        comment="Tên model dùng để tạo embedding, VD: w600k_r50"
    )
    model_version: Mapped[str] = mapped_column(
        String(50), nullable=False, default="1.0",
        comment="Phiên bản model — dùng để detect khi cần re-enroll"
    )
    quality_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Điểm chất lượng khuôn mặt lúc đăng ký (0.0–1.0)"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
        comment="True = dùng để so khớp; False = đã vô hiệu hóa"
    )
    enrolled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Thời điểm hết hạn; NULL = không hết hạn"
    )

    # Relationship
    employee: Mapped[Employee] = relationship("Employee", back_populates="face_templates")

    # Indexes
    __table_args__ = (
        Index("ix_face_templates_employee_active", "employee_id", "is_active"),
        Index("ix_face_templates_model", "model_name", "model_version"),
    )

    def get_embedding(self) -> np.ndarray:
        """Giải mã embedding từ binary về numpy array (512,)."""
        return bytes_to_embedding(self.embedding)

    def set_embedding(self, arr: np.ndarray) -> None:
        """Mã hóa và lưu numpy array vào cột embedding."""
        self.embedding = embedding_to_bytes(arr)

    def __repr__(self) -> str:
        # KHÔNG hiển thị embedding trong repr
        return (
            f"<FaceTemplate id={self.id} employee_id={self.employee_id} "
            f"model={self.model_name}@{self.model_version} active={self.is_active}>"
        )


# ---------------------------------------------------------------------------
# Model: cameras
# ---------------------------------------------------------------------------

class Camera(Base):
    """Thông tin và cấu hình camera."""
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_code: Mapped[str] = mapped_column(
        String(50), nullable=False, unique=True,
        comment="Mã camera duy nhất (VD: CAM001)"
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    stream_url: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True,
        comment="rtsp://... hoặc số thứ tự webcam (VD: '0'). Không lưu password ở đây."
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="online",
        comment="online | offline | disabled"
    )
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationship
    attendance_events: Mapped[list["AttendanceEvent"]] = relationship(
        "AttendanceEvent", back_populates="camera"
    )
    face_recognition_logs: Mapped[list["FaceRecognitionLog"]] = relationship(
        "FaceRecognitionLog", back_populates="camera"
    )

    def __repr__(self) -> str:
        return f"<Camera id={self.id} code={self.camera_code!r} location={self.location!r} status={self.status}>"


# ---------------------------------------------------------------------------
# Model: attendance_events
# ---------------------------------------------------------------------------

class AttendanceEvent(Base):
    """
    Sự kiện chấm công (check-in / check-out).

    Timestamp luôn lưu dạng UTC (timezone-aware).
    Hiển thị theo GMT+7 ở tầng ứng dụng / API.
    """
    __tablename__ = "attendance_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("employees.id", ondelete="RESTRICT"), nullable=False,
        comment="Không xóa cascade — giữ lịch sử dù nhân viên bị xóa"
    )
    camera_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(
        String(20), nullable=False,
        comment="check_in | check_out"
    )
    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        comment="Thời điểm sự kiện — lưu UTC"
    )
    confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Điểm cosine similarity lúc nhận diện (0.0–1.0)"
    )
    verify_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="auto",
        comment="auto | manual | disputed"
    )
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    employee: Mapped[Employee] = relationship("Employee", back_populates="attendance_events")
    camera: Mapped[Optional[Camera]] = relationship("Camera", back_populates="attendance_events")

    # Indexes để query lịch sử nhanh
    __table_args__ = (
        Index("ix_attendance_employee_time", "employee_id", "event_time"),
        Index("ix_attendance_event_time", "event_time"),
        Index("ix_attendance_camera_time", "camera_id", "event_time"),
    )

    def __repr__(self) -> str:
        return (
            f"<AttendanceEvent id={self.id} employee_id={self.employee_id} "
            f"type={self.event_type!r} time={self.event_time} status={self.verify_status}>"
        )


# ---------------------------------------------------------------------------
# Model: audit_logs
# ---------------------------------------------------------------------------

class AuditLog(Base):
    """
    Nhật ký thao tác quan trọng (ai sửa gì, khi nào).

    KHÔNG lưu embedding thô vào old_value / new_value.
    Chỉ lưu metadata (id, trạng thái, thời gian).
    """
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
        comment="NULL nếu action từ system/scheduler"
    )
    action: Mapped[str] = mapped_column(
        String(20), nullable=False,
        comment="CREATE | UPDATE | DELETE | VIEW"
    )
    target_table: Mapped[str] = mapped_column(String(100), nullable=False)
    target_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    old_value: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="JSON snapshot trước khi sửa — KHÔNG chứa embedding"
    )
    new_value: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="JSON snapshot sau khi sửa — KHÔNG chứa embedding"
    )
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationship
    user: Mapped[Optional[User]] = relationship("User", back_populates="audit_logs")

    # Index
    __table_args__ = (
        Index("ix_audit_target", "target_table", "target_id"),
        Index("ix_audit_created", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} action={self.action!r} "
            f"table={self.target_table!r} target_id={self.target_id} user_id={self.user_id}>"
        )


# ---------------------------------------------------------------------------
# Model: face_recognition_logs  (Log kỹ thuật — Giai đoạn 10)
# ---------------------------------------------------------------------------

class FaceRecognitionLog(Base):
    """
    Log kỹ thuật — ghi lại MỖI LẦN camera thực hiện nhận diện khuôn mặt.

    Khác với attendance_events (chỉ ghi sự kiện được chấp nhận),
    bảng này ghi TẤT CẢ: thành công, thất bại, uncertainty.

    Chính sách lưu trữ đề xuất:
      - Kết quả SUCCESS  : giữ 30–90 ngày
      - Kết quả FAILURE  : giữ 7–30 ngày
      - Ảnh sự kiện      : giữ 7–30 ngày nếu thực sự cần, có chính sách bảo vệ
    """
    __tablename__ = "face_recognition_logs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="LogId — tăng dần, VD: 10001"
    )
    employee_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("employees.id", ondelete="SET NULL"), nullable=True,
        comment="NULL nếu nhận diện thất bại (UNKNOWN / UNCERTAIN)"
    )
    camera_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        comment="Thời điểm nhận diện — lưu UTC (VD: 08:00:15)"
    )
    confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Cosine similarity / confidence score (0.0–1.0), VD: 0.95"
    )
    result: Mapped[str] = mapped_column(
        String(20), nullable=False,
        comment="SUCCESS | FAILURE | UNCERTAIN"
    )
    failure_reason: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
        comment="Lý do thất bại nếu result != SUCCESS, VD: LOW_QUALITY, UNKNOWN_FACE"
    )
    event_snapshot_path: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True,
        comment="Đường dẫn ảnh chụp sự kiện (nếu bật chụp). "
                "Chỉ lưu khi thực sự cần và có chính sách bảo vệ phù hợp. "
                "Xóa sau 7–30 ngày theo policy."
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Thời điểm hết hạn lưu trữ (scheduler sẽ xóa sau thời hạn này)"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    employee: Mapped[Optional["Employee"]] = relationship(
        "Employee", back_populates="face_recognition_logs"
    )
    camera: Mapped[Optional["Camera"]] = relationship(
        "Camera", back_populates="face_recognition_logs"
    )

    # Indexes
    __table_args__ = (
        Index("ix_frl_timestamp", "timestamp"),
        Index("ix_frl_employee_time", "employee_id", "timestamp"),
        Index("ix_frl_result", "result"),
        Index("ix_frl_expires_at", "expires_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<FaceRecognitionLog id={self.id} emp={self.employee_id} "
            f"result={self.result!r} ts={self.timestamp} conf={self.confidence}>"
        )


# ---------------------------------------------------------------------------
# Model: shifts  (Ca làm việc — Giai đoạn 11)
# ---------------------------------------------------------------------------

class Shift(Base):
    """
    Cấu hình ca làm việc (ví dụ: Ca hành chính 08:00–17:00, Ca đêm 22:00–06:00).

    Hỗ trợ:
      - Nghỉ trưa/nghỉ giữa ca theo khung giờ (break_start – break_end)
      - Grace period (thời gian ân hạn đi trễ)
      - Ca qua ngày / ca đêm (is_overnight=True)
      - Khung giờ hợp lệ để Check-in sớm và Check-out muộn
    """
    __tablename__ = "shifts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shift_code: Mapped[str] = mapped_column(
        String(50), nullable=False, unique=True,
        comment="Mã ca duy nhất, VD: HC_08_17, CA_DEM_22_06"
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    start_time: Mapped[time_type] = mapped_column(
        Time, nullable=False, comment="Giờ bắt đầu ca (local time), VD: 08:00:00"
    )
    end_time: Mapped[time_type] = mapped_column(
        Time, nullable=False, comment="Giờ kết thúc ca (local time), VD: 17:00:00"
    )
    break_start: Mapped[Optional[time_type]] = mapped_column(
        Time, nullable=True, comment="Giờ bắt đầu nghỉ giữa ca, VD: 12:00:00"
    )
    break_end: Mapped[Optional[time_type]] = mapped_column(
        Time, nullable=True, comment="Giờ kết thúc nghỉ giữa ca, VD: 13:00:00"
    )
    break_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=60,
        comment="Thời gian nghỉ tiêu chuẩn theo lịch (phút)"
    )
    standard_work_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=480,
        comment="Thời gian làm việc tiêu chuẩn của ca (phút, mặc định 8h = 480p)"
    )
    grace_period_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5,
        comment="Thời gian ân hạn đi trễ (phút)"
    )
    early_check_in_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=120,
        comment="Cho phép check-in sớm tối đa trước giờ bắt đầu ca (phút)"
    )
    late_check_out_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=240,
        comment="Cho phép check-out muộn tối đa sau giờ kết thúc ca (phút)"
    )
    is_overnight: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        comment="True nếu ca làm qua nửa đêm (VD: 22:00 -> 06:00 hôm sau)"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    attendance_records: Mapped[list["AttendanceRecord"]] = relationship(
        "AttendanceRecord", back_populates="shift"
    )

    def __repr__(self) -> str:
        return (
            f"<Shift id={self.id} code={self.shift_code!r} "
            f"{self.start_time}-{self.end_time} overnight={self.is_overnight}>"
        )


# ---------------------------------------------------------------------------
# Model: attendance_records  (Bản ghi chính thức — Giai đoạn 10/11)
# ---------------------------------------------------------------------------

class AttendanceRecord(Base):
    """
    Bản ghi chấm công CHÍNH THỨC theo ca / ngày làm việc.

    Trạng thái nghiệp vụ bắt buộc (attendance_status):
      - IN_PROGRESS : Đã Check-in nhưng chưa Check-out
      - COMPLETED   : Đã có cả Check-in và Check-out
      - INCOMPLETE  : Thiếu Check-in hoặc Check-out sau khi ca kết thúc và đã đối soát

    Chỉ số thời gian:
      - total_presence_minutes = CheckOutTime - CheckInTime
      - break_minutes          = Thời gian nghỉ thực tế trong ca (theo lịch)
      - actual_work_minutes    = TotalPresenceMinutes - BreakMinutes
      - standard_work_minutes  = Thời gian tiêu chuẩn của ca
      - missing_minutes        = MAX(0, StandardWorkMinutes - ActualWorkMinutes)
      - extra_minutes          = MAX(0, ActualWorkMinutes - StandardWorkMinutes)
      - overtime_minutes       = Chỉ ghi nhận khi có phê duyệt tăng ca (overtime_approved=True)
    """
    __tablename__ = "attendance_records"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="AttendanceId — tăng dần, VD: 10001"
    )
    employee_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("employees.id", ondelete="RESTRICT"), nullable=False,
        comment="Không xóa cascade — giữ lịch sử dù nhân viên bị xóa"
    )
    shift_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("shifts.id", ondelete="SET NULL"), nullable=True,
        comment="FK đến ca làm việc (shifts)"
    )
    work_date: Mapped["date_type"] = mapped_column(
        Date, nullable=False,
        comment="Ngày làm việc (YYYY-MM-DD). Phân cấp: năm → tháng → ngày"
    )

    # --- Thông tin check-in ---
    check_in: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Thời điểm check-in — UTC (CheckInTime)"
    )
    check_in_camera_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    check_in_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Confidence khi check-in (0.0–1.0)"
    )

    # --- Thông tin check-out ---
    check_out: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Thời điểm check-out — UTC (CheckOutTime)"
    )
    check_out_camera_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    check_out_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Confidence khi check-out (0.0–1.0)"
    )

    # --- Trạng thái điểm danh ---
    attendance_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="IN_PROGRESS",
        comment="IN_PROGRESS | COMPLETED | INCOMPLETE"
    )
    punctuality_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="ON_TIME",
        comment="Trạng thái đúng/trễ/về sớm: ON_TIME | LATE | EARLY_LEAVE | LATE_AND_EARLY | UNKNOWN"
    )
    late_minutes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=0,
        comment="LateMinutes — số phút đi trễ"
    )
    early_leave_minutes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=0,
        comment="EarlyLeaveMinutes — số phút về sớm"
    )
    total_presence_minutes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
        comment="TotalPresenceMinutes = CheckOutTime - CheckInTime"
    )
    break_minutes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=0,
        comment="BreakMinutes — thời gian nghỉ theo lịch được trừ"
    )
    actual_work_minutes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
        comment="ActualWorkMinutes = TotalPresenceMinutes - BreakMinutes"
    )
    standard_work_minutes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=480,
        comment="StandardWorkMinutes — thời gian làm việc tiêu chuẩn"
    )
    missing_minutes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=0,
        comment="MissingMinutes = MAX(0, StandardWorkMinutes - ActualWorkMinutes)"
    )
    extra_minutes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=0,
        comment="ExtraMinutes = MAX(0, ActualWorkMinutes - StandardWorkMinutes)"
    )
    overtime_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="OvertimeMinutes — chỉ ghi nhận khi có phê duyệt tăng ca hợp lệ"
    )
    overtime_approved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        comment="True nếu tăng ca đã được HR/Quản lý phê duyệt"
    )
    work_minutes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
        comment="Alias tương thích ngược cho actual_work_minutes"
    )

    # --- Đối soát ngoại lệ (HR / Quản lý kiểm tra) ---
    needs_review: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        comment="True nếu thiếu Check-in / Check-out hoặc cần HR đối soát"
    )
    review_reason: Mapped[Optional[str]] = mapped_column(
        String(200), nullable=True,
        comment="VD: MISSING_CHECK_IN | MISSING_CHECK_OUT | OUTSIDE_SHIFT"
    )

    # --- Thông tin ca làm việc ---
    shift_start: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Giờ bắt đầu ca quy định — UTC"
    )
    shift_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Giờ kết thúc ca quy định — UTC"
    )

    # --- Phương thức và xác minh ---
    verify_method: Mapped[str] = mapped_column(
        String(50), nullable=False, default="Face Recognition",
        comment="Phương thức điểm danh: Face Recognition | Manual | Card"
    )
    verify_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="Success",
        comment="Success | Failed | Disputed"
    )
    result: Mapped[str] = mapped_column(
        String(20), nullable=False, default="Accepted",
        comment="Accepted | Rejected | Pending"
    )
    device_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
        comment="ID thiết bị/camera ghi nhận"
    )
    record_source: Mapped[str] = mapped_column(
        String(20), nullable=False, default="auto",
        comment="auto (nhận diện tự động) | manual (admin nhập tay) | corrected (đã chỉnh)"
    )

    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    employee: Mapped["Employee"] = relationship(
        "Employee", back_populates="attendance_records"
    )
    shift: Mapped[Optional["Shift"]] = relationship(
        "Shift", back_populates="attendance_records"
    )
    check_in_camera: Mapped[Optional["Camera"]] = relationship(
        "Camera", foreign_keys=[check_in_camera_id]
    )
    check_out_camera: Mapped[Optional["Camera"]] = relationship(
        "Camera", foreign_keys=[check_out_camera_id]
    )

    # Indexes — phân cấp theo năm/tháng/ngày (work_date)
    __table_args__ = (
        # Đảm bảo mỗi nhân viên chỉ có 1 bản ghi chính thức mỗi ngày
        UniqueConstraint("employee_id", "work_date", name="uq_attendance_emp_date"),
        # Query nhanh theo ngày làm việc (toàn công ty)
        Index("ix_attendance_records_work_date", "work_date"),
        # Query lịch sử theo nhân viên + ngày
        Index("ix_attendance_records_emp_date", "employee_id", "work_date"),
        # Query theo trạng thái
        Index("ix_attendance_records_status", "attendance_status"),
        Index("ix_attendance_records_punctuality", "punctuality_status"),
        Index("ix_attendance_records_needs_review", "needs_review"),
    )

    def to_display_dict(
        self,
        tz_offset_hours: int = 7,
        employee_code: Optional[str] = None,
        event_type: str = "Check-in",
    ) -> dict:
        """
        Trả về dict hiển thị theo cấu trúc nghiệp vụ chuẩn.
        Dữ liệu hiển thị theo múi giờ local (mặc định GMT+7).
        """
        from datetime import timedelta
        tz = timezone(timedelta(hours=tz_offset_hours))

        def _fmt(dt: Optional[datetime]) -> Optional[str]:
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S")

        emp_display = (
            employee_code
            or (self.employee.employee_code if self.employee is not None else str(self.employee_id))
        )
        event_dt = self.check_out if event_type.lower() == "check-out" and self.check_out else self.check_in

        return {
            "Record ID":            self.id,
            "Employee ID":          emp_display,
            "Shift ID":             self.shift_id,
            "Work Date":            str(self.work_date),
            "Thời gian":            _fmt(event_dt),
            "Device ID":            self.device_id,
            "Phương thức":          self.verify_method,
            "Event Type":           event_type,
            "Verify Status":        self.verify_status,
            "Result":               self.result,
            "Check-in":             _fmt(self.check_in),
            "Check-out":            _fmt(self.check_out),
            "Trạng thái":           self.attendance_status,
            "Punctuality":          self.punctuality_status,
            "Đi trễ (phút)":        self.late_minutes,
            "Về sớm (phút)":        self.early_leave_minutes,
            "Có mặt (phút)":        self.total_presence_minutes,
            "Nghỉ (phút)":          self.break_minutes,
            "Làm (phút)":           self.actual_work_minutes if self.actual_work_minutes is not None else self.work_minutes,
            "Tiêu chuẩn (phút)":    self.standard_work_minutes,
            "Thiếu (phút)":         self.missing_minutes,
            "Vượt giờ (phút)":      self.extra_minutes,
            "Tăng ca duyệt (phút)": self.overtime_minutes,
            "Cần đối soát":         self.needs_review,
        }

    def __repr__(self) -> str:
        return (
            f"<AttendanceRecord id={self.id} emp={self.employee_id} "
            f"date={self.work_date} status={self.attendance_status!r} "
            f"punctuality={self.punctuality_status!r}>"
        )
