"""
app/database/db_sync.py
Lớp cầu nối đồng bộ giữa GUI / AI Pipeline và PostgreSQL (faceattend_db).

Thay thế hoàn toàn lưu trữ file cục bộ cũ (gallery.npz, employee_counter.json,
registered_faces/, SQLite fallback) bằng PostgreSQL qua SQLAlchemy repositories.
"""

from __future__ import annotations

import logging
import time as time_mod
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sqlalchemy import select

from app.attendance.attendance_service import AttendanceDecision, EventType
from app.database.connection import get_session
from app.database.models import Camera, Employee
from app.database.repositories import (
    AttendanceRecordRepository,
    AttendanceRepository,
    EmployeeRepository,
    FaceRecognitionLogRepository,
    FaceTemplateRepository,
)
from app.vision.face_matcher import FaceGallery

logger = logging.getLogger(__name__)

MODEL_NAME = "w600k_r50"
MODEL_VERSION = "1.0"


def _get_or_create_camera(session, camera_code: str) -> Camera:
    """Tìm camera theo camera_code, tự tạo mới nếu chưa có."""
    cam = session.execute(
        select(Camera).where(Camera.camera_code == camera_code)
    ).scalar_one_or_none()
    if cam is None:
        cam = Camera(
            camera_code=camera_code,
            name=f"Camera {camera_code}",
            stream_url="0",
            status="online",
        )
        session.add(cam)
        session.flush()
    return cam


def load_gallery_from_db(
    model_name: str = MODEL_NAME,
    model_version: str = MODEL_VERSION,
) -> FaceGallery:
    """Nạp toàn bộ nhân viên active và embedding từ PostgreSQL vào FaceGallery."""
    gallery = FaceGallery()
    with get_session() as session:
        emp_repo = EmployeeRepository(session)
        tpl_repo = FaceTemplateRepository(session)

        active_emps = emp_repo.list_active()
        emp_by_id = {e.id: e for e in active_emps}
        raw_gallery = tpl_repo.load_gallery(
            model_name=model_name,
            model_version=model_version,
            only_active_employees=True,
        )
        for emp_pk, embeddings in raw_gallery.items():
            emp = emp_by_id.get(emp_pk)
            if emp is None:
                continue
            for emb in embeddings:
                gallery.add(emp.employee_code, emb, name=emp.full_name)

    logger.info("Đã nạp gallery từ PostgreSQL: %d nhân viên (%d mẫu).", gallery.size, gallery.total_samples)
    return gallery


def generate_next_employee_code_from_db(prefix: str = "FA") -> str:
    """Sinh mã nhân viên tiếp theo dạng <PREFIX>-<YYYY>-<NNNN> dựa trên DB."""
    year = time_mod.strftime("%Y")
    code_prefix = f"{prefix.upper()}-{year}-"
    max_seq = 0
    with get_session() as session:
        codes = session.execute(
            select(Employee.employee_code).where(Employee.employee_code.like(f"{code_prefix}%"))
        ).scalars().all()
        for code in codes:
            suffix = code[len(code_prefix):]
            if suffix.isdigit():
                max_seq = max(max_seq, int(suffix))
    return f"{code_prefix}{max_seq + 1:04d}"


def save_enrolled_employee_to_db(
    employee_code: str,
    full_name: str,
    department: str,
    embeddings: list[np.ndarray],
    model_name: str = MODEL_NAME,
    model_version: str = MODEL_VERSION,
) -> int:
    """Lưu nhân viên mới và các embedding mẫu vào PostgreSQL. Trả về employee PK."""
    with get_session() as session:
        emp_repo = EmployeeRepository(session)
        tpl_repo = FaceTemplateRepository(session)

        emp = emp_repo.get_by_code(employee_code)
        if emp is None:
            emp = emp_repo.create(
                employee_code=employee_code,
                full_name=full_name,
                department=department or None,
            )
        else:
            emp_repo.update(emp.id, full_name=full_name, department=department or None)
            if emp.status != "active":
                emp_repo.set_status(emp.id, "active")

        for emb in embeddings:
            tpl_repo.add(
                employee_id=emp.id,
                embedding=emb,
                model_name=model_name,
                model_version=model_version,
            )
        return emp.id


def delete_employee_from_db(employee_code: str) -> bool:
    """Đánh dấu xóa (soft-delete) nhân viên và vô hiệu hóa toàn bộ template trong PostgreSQL."""
    with get_session() as session:
        emp_repo = EmployeeRepository(session)
        tpl_repo = FaceTemplateRepository(session)

        emp = emp_repo.get_by_code(employee_code)
        if emp is None:
            return False

        for tpl in tpl_repo.list_active_by_employee(emp.id):
            tpl_repo.deactivate(tpl.id)
        emp_repo.set_status(emp.id, "deleted")
        return True


def record_attendance_to_db(decision: AttendanceDecision, camera_code: str = "cam_0") -> None:
    """Ghi sự kiện chấm công (attendance_events + attendance_records) vào PostgreSQL."""
    if not decision.created or decision.event is None:
        return

    ev = decision.event
    event_dt = datetime.fromtimestamp(ev.timestamp_utc, tz=timezone.utc)

    with get_session() as session:
        emp_repo = EmployeeRepository(session)
        att_repo = AttendanceRepository(session)
        rec_repo = AttendanceRecordRepository(session)

        emp = emp_repo.get_by_code(ev.employee_id)
        if emp is None:
            logger.warning("record_attendance_to_db: Không tìm thấy employee_code=%s trong DB.", ev.employee_id)
            return

        cam = _get_or_create_camera(session, camera_code)
        att_repo.record_event(
            employee_id=emp.id,
            event_type=ev.event_type.value,
            event_time=event_dt,
            camera_id=cam.id,
            confidence=ev.similarity,
            verify_status="auto",
        )

        rec_state = decision.record
        if ev.event_type == EventType.CHECK_IN:
            rec_repo.upsert_check_in(
                employee_id=emp.id,
                check_in=event_dt,
                camera_id=cam.id,
                confidence=ev.similarity,
                punctuality_status=ev.punctuality_status.value,
                late_minutes=ev.late_minutes or 0,
                shift_start=rec_state.shift_start_utc if rec_state else None,
                shift_end=rec_state.shift_end_utc if rec_state else None,
                standard_work_minutes=rec_state.standard_work_minutes if rec_state else 480,
                device_id=camera_code,
                verify_method=ev.verify_method,
                work_date_override=ev.work_date,
            )
        else:
            updated = rec_repo.upsert_check_out(
                employee_id=emp.id,
                check_out=event_dt,
                camera_id=cam.id,
                confidence=ev.similarity,
                early_leave_minutes=ev.early_leave_minutes or 0,
                break_minutes=rec_state.break_minutes if rec_state else 0,
                standard_work_minutes=rec_state.standard_work_minutes if rec_state else 480,
                work_date_override=ev.work_date,
            )
            if updated is None:
                rec_repo.record_missing_checkin_checkout(
                    employee_id=emp.id,
                    check_out=event_dt,
                    camera_id=cam.id,
                    confidence=ev.similarity,
                    early_leave_minutes=ev.early_leave_minutes or 0,
                    standard_work_minutes=rec_state.standard_work_minutes if rec_state else 480,
                    work_date_override=ev.work_date,
                )


def log_recognition_to_db(
    result: str,
    employee_code: Optional[str] = None,
    camera_code: str = "cam_0",
    confidence: Optional[float] = None,
    failure_reason: Optional[str] = None,
) -> None:
    """Ghi log kỹ thuật mỗi lần nhận diện vào bảng face_recognition_logs."""
    with get_session() as session:
        emp_repo = EmployeeRepository(session)
        log_repo = FaceRecognitionLogRepository(session)

        emp_pk: Optional[int] = None
        if employee_code:
            emp = emp_repo.get_by_code(employee_code)
            if emp is not None:
                emp_pk = emp.id

        cam = _get_or_create_camera(session, camera_code)
        log_repo.record_recognition(
            timestamp=datetime.now(timezone.utc),
            result=result,
            employee_id=emp_pk,
            camera_id=cam.id,
            confidence=confidence,
            failure_reason=failure_reason,
        )
