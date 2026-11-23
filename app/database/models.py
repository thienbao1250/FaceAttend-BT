"""
app/database/models.py
SQLAlchemy ORM models — Giai đoạn 9.

Ánh xạ trực tiếp từ ERD đã thiết kế:
  - employees          : thông tin nhân viên
  - face_templates     : embedding khuôn mặt (binary bytea/BLOB)
  - cameras            : thông tin camera
  - attendance_events  : sự kiện check-in / check-out
  - users              : tài khoản quản trị
  - roles              : vai trò và quyền
  - audit_logs         : nhật ký thao tác

Lưu ý bảo mật:
  - Cột `embedding` lưu dạng LargeBinary (bytea/BLOB) — không log giá trị này.
  - Cột `password_hash` trong users — không log giá trị này.
  - audit_logs.old_value / new_value KHÔNG được chứa embedding thô.
"""

from __future__ import annotations

import struct
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Index, Integer, LargeBinary, String, Text, UniqueConstraint,
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

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
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
