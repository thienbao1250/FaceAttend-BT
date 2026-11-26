"""
app/database/repositories/attendance_record_repo.py
Data-access layer cho bảng attendance_records — Giai đoạn 10.

Bảng attendance_records là bảng CHÍNH THỨC để tính công.
Mỗi nhân viên chỉ có MỘT bản ghi mỗi ngày (UniqueConstraint employee_id + work_date).
Dữ liệu được phân cấp theo work_date: năm → tháng → ngày.
"""

from __future__ import annotations

import logging
from datetime import date as date_type, datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, and_, extract, func
from sqlalchemy.orm import Session

from app.database.models import AttendanceRecord

logger = logging.getLogger(__name__)


def _ensure_utc(dt: datetime) -> datetime:
    """Đảm bảo datetime luôn có tzinfo=UTC (SQLite có thể strip tzinfo khi đọc lại)."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


class AttendanceRecordRepository:
    """CRUD operations cho bảng attendance_records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Create / Upsert
    # ------------------------------------------------------------------

    def upsert_check_in(
        self,
        employee_id: int,
        check_in: datetime,
        camera_id: Optional[int] = None,
        confidence: Optional[float] = None,
        punctuality_status: str = "ON_TIME",
        late_minutes: Optional[int] = None,
        shift_start: Optional[datetime] = None,
        shift_end: Optional[datetime] = None,
        device_id: Optional[str] = None,
        verify_method: str = "Face Recognition",
        tz_offset_hours: int = 7,
    ) -> AttendanceRecord:
        """
        Ghi / cập nhật check-in cho nhân viên trong ngày.

        Nếu bản ghi ngày hôm đó đã tồn tại:
          - Chỉ cập nhật nếu check_in mới sớm hơn (lần đầu vào)
        Nếu chưa có: tạo mới.

        Dữ liệu phân cấp theo work_date: năm → tháng → ngày.
        """
        if check_in.tzinfo is None:
            check_in = check_in.replace(tzinfo=timezone.utc)

        # Xác định ngày làm việc theo múi giờ local
        tz = timezone(timedelta(hours=tz_offset_hours))
        work_date = check_in.astimezone(tz).date()

        # Tính trạng thái chính
        attendance_status = "PRESENT"
        verify_status = "Success"
        result_val = "Accepted"

        record = self._get_by_emp_date(employee_id, work_date)

        if record is None:
            record = AttendanceRecord(
                employee_id=employee_id,
                work_date=work_date,
                check_in=check_in,
                check_in_camera_id=camera_id,
                check_in_confidence=confidence,
                attendance_status=attendance_status,
                punctuality_status=punctuality_status,
                late_minutes=late_minutes,
                shift_start=shift_start,
                shift_end=shift_end,
                device_id=device_id,
                verify_method=verify_method,
                verify_status=verify_status,
                result=result_val,
            )
            self._session.add(record)
        else:
            # Giữ lần check-in sớm nhất
            existing_ci = _ensure_utc(record.check_in) if record.check_in is not None else None
            if existing_ci is None or check_in < existing_ci:
                record.check_in = check_in
                record.check_in_camera_id = camera_id
                record.check_in_confidence = confidence
                record.punctuality_status = punctuality_status
                record.late_minutes = late_minutes
                if device_id:
                    record.device_id = device_id

        self._session.flush()
        logger.info(
            "AttendanceRecord check-in: emp=%d date=%s status=%s late=%s",
            employee_id, work_date, punctuality_status, late_minutes,
        )
        return record

    def upsert_check_out(
        self,
        employee_id: int,
        check_out: datetime,
        camera_id: Optional[int] = None,
        confidence: Optional[float] = None,
        early_leave_minutes: Optional[int] = None,
        tz_offset_hours: int = 7,
    ) -> Optional[AttendanceRecord]:
        """
        Ghi / cập nhật check-out cho nhân viên trong ngày.
        Giữ lần check-out muộn nhất (ra lần cuối).
        Tính work_minutes = check_out - check_in.
        """
        if check_out.tzinfo is None:
            check_out = check_out.replace(tzinfo=timezone.utc)

        tz = timezone(timedelta(hours=tz_offset_hours))
        work_date = check_out.astimezone(tz).date()

        record = self._get_by_emp_date(employee_id, work_date)
        if record is None:
            logger.warning(
                "Không tìm thấy attendance_record check-in cho emp=%d date=%s. "
                "Bỏ qua check-out.", employee_id, work_date
            )
            return None

        # Giữ lần check-out muộn nhất
        existing_co = _ensure_utc(record.check_out) if record.check_out is not None else None
        if existing_co is None or check_out > existing_co:
            record.check_out = check_out
            record.check_out_camera_id = camera_id
            record.check_out_confidence = confidence

        # Tính tổng thời gian làm việc
        if record.check_in and record.check_out:
            diff = _ensure_utc(record.check_out) - _ensure_utc(record.check_in)
            record.work_minutes = max(0, int(diff.total_seconds() / 60))

        # Cập nhật về sớm nếu có
        if early_leave_minutes is not None:
            record.early_leave_minutes = early_leave_minutes

        self._session.flush()
        logger.info(
            "AttendanceRecord check-out: emp=%d date=%s work_min=%s",
            employee_id, work_date, record.work_minutes,
        )
        return record

    # ------------------------------------------------------------------
    # Read — Phân cấp theo năm / tháng / ngày
    # ------------------------------------------------------------------

    def _get_by_emp_date(
        self, employee_id: int, work_date: date_type
    ) -> Optional[AttendanceRecord]:
        """Lấy bản ghi chấm công của một nhân viên trong một ngày."""
        stmt = select(AttendanceRecord).where(
            and_(
                AttendanceRecord.employee_id == employee_id,
                AttendanceRecord.work_date == work_date,
            )
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_by_id(self, record_id: int) -> Optional[AttendanceRecord]:
        return self._session.get(AttendanceRecord, record_id)

    def list_by_employee(
        self,
        employee_id: int,
        date_from: Optional[date_type] = None,
        date_to: Optional[date_type] = None,
        limit: int = 100,
    ) -> list[AttendanceRecord]:
        """
        Lịch sử chấm công của một nhân viên theo khoảng ngày.
        Phân cấp: lọc theo work_date (năm → tháng → ngày).
        """
        stmt = (
            select(AttendanceRecord)
            .where(AttendanceRecord.employee_id == employee_id)
            .order_by(AttendanceRecord.work_date.desc())
            .limit(limit)
        )
        if date_from:
            stmt = stmt.where(AttendanceRecord.work_date >= date_from)
        if date_to:
            stmt = stmt.where(AttendanceRecord.work_date <= date_to)
        return list(self._session.execute(stmt).scalars().all())

    def list_by_date(self, work_date: date_type) -> list[AttendanceRecord]:
        """Lấy tất cả bản ghi chấm công trong một ngày cụ thể."""
        stmt = (
            select(AttendanceRecord)
            .where(AttendanceRecord.work_date == work_date)
            .order_by(AttendanceRecord.check_in)
        )
        return list(self._session.execute(stmt).scalars().all())

    def list_by_month(
        self, year: int, month: int
    ) -> list[AttendanceRecord]:
        """
        Lấy tất cả bản ghi trong một tháng.
        Phân cấp: năm → tháng → (mỗi ngày).
        """
        stmt = (
            select(AttendanceRecord)
            .where(
                and_(
                    extract("year", AttendanceRecord.work_date) == year,
                    extract("month", AttendanceRecord.work_date) == month,
                )
            )
            .order_by(AttendanceRecord.work_date, AttendanceRecord.employee_id)
        )
        return list(self._session.execute(stmt).scalars().all())

    def list_by_year(self, year: int) -> list[AttendanceRecord]:
        """
        Lấy tất cả bản ghi trong một năm.
        Phân cấp: năm → (mỗi tháng → mỗi ngày).
        """
        stmt = (
            select(AttendanceRecord)
            .where(extract("year", AttendanceRecord.work_date) == year)
            .order_by(AttendanceRecord.work_date, AttendanceRecord.employee_id)
        )
        return list(self._session.execute(stmt).scalars().all())

    def list_late_records(
        self,
        date_from: Optional[date_type] = None,
        date_to: Optional[date_type] = None,
        employee_id: Optional[int] = None,
    ) -> list[AttendanceRecord]:
        """Lấy danh sách bản ghi đi trễ."""
        stmt = (
            select(AttendanceRecord)
            .where(AttendanceRecord.punctuality_status == "LATE")
            .order_by(AttendanceRecord.work_date.desc())
        )
        if date_from:
            stmt = stmt.where(AttendanceRecord.work_date >= date_from)
        if date_to:
            stmt = stmt.where(AttendanceRecord.work_date <= date_to)
        if employee_id is not None:
            stmt = stmt.where(AttendanceRecord.employee_id == employee_id)
        return list(self._session.execute(stmt).scalars().all())

    def count_late_days(
        self,
        employee_id: int,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> int:
        """Đếm số ngày đi trễ của một nhân viên (tùy chọn theo tháng/năm)."""
        stmt = select(func.count()).where(
            and_(
                AttendanceRecord.employee_id == employee_id,
                AttendanceRecord.punctuality_status == "LATE",
            )
        )
        if year:
            stmt = stmt.where(extract("year", AttendanceRecord.work_date) == year)
        if month:
            stmt = stmt.where(extract("month", AttendanceRecord.work_date) == month)
        return self._session.execute(stmt).scalar_one() or 0

    def count_present_days(
        self,
        employee_id: int,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> int:
        """Đếm số ngày có mặt của một nhân viên."""
        stmt = select(func.count()).where(
            and_(
                AttendanceRecord.employee_id == employee_id,
                AttendanceRecord.attendance_status == "PRESENT",
            )
        )
        if year:
            stmt = stmt.where(extract("year", AttendanceRecord.work_date) == year)
        if month:
            stmt = stmt.where(extract("month", AttendanceRecord.work_date) == month)
        return self._session.execute(stmt).scalar_one() or 0

    # ------------------------------------------------------------------
    # Update (admin chỉnh sửa thủ công)
    # ------------------------------------------------------------------

    def update_status(
        self,
        record_id: int,
        attendance_status: Optional[str] = None,
        punctuality_status: Optional[str] = None,
        note: Optional[str] = None,
        record_source: str = "corrected",
    ) -> AttendanceRecord:
        """Cập nhật trạng thái bản ghi (admin chỉnh thủ công)."""
        record = self.get_by_id(record_id)
        if record is None:
            raise ValueError(f"Không tìm thấy attendance_record id={record_id}")

        valid_att = {"PRESENT", "ABSENT", "LATE", "HALF_DAY", "HOLIDAY", "LEAVE"}
        valid_punc = {"ON_TIME", "LATE", "EARLY_LEAVE", "UNKNOWN"}

        if attendance_status is not None:
            if attendance_status not in valid_att:
                raise ValueError(f"attendance_status không hợp lệ: {attendance_status!r}")
            record.attendance_status = attendance_status

        if punctuality_status is not None:
            if punctuality_status not in valid_punc:
                raise ValueError(f"punctuality_status không hợp lệ: {punctuality_status!r}")
            record.punctuality_status = punctuality_status

        if note is not None:
            record.note = note

        record.record_source = record_source
        self._session.flush()
        logger.info(
            "AttendanceRecord updated: id=%d att=%s punc=%s",
            record_id, record.attendance_status, record.punctuality_status,
        )
        return record
