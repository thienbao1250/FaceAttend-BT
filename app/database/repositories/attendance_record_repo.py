"""
app/database/repositories/attendance_record_repo.py
Data-access layer cho bảng attendance_records — Giai đoạn 10/11.

Bảng attendance_records là bảng CHÍNH THỨC để tính công.
Mỗi nhân viên chỉ có MỘT bản ghi mỗi ngày/ca (UniqueConstraint employee_id + work_date).

Trạng thái nghiệp vụ bắt buộc (attendance_status):
  - IN_PROGRESS : Đã Check-in nhưng chưa Check-out
  - COMPLETED   : Đã có cả Check-in và Check-out
  - INCOMPLETE  : Thiếu Check-in hoặc Check-out sau khi ca kết thúc và đã đối soát
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
    """CRUD & đối soát nghiệp vụ cho bảng attendance_records."""

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
        late_minutes: Optional[int] = 0,
        shift_id: Optional[int] = None,
        shift_start: Optional[datetime] = None,
        shift_end: Optional[datetime] = None,
        standard_work_minutes: int = 480,
        device_id: Optional[str] = None,
        verify_method: str = "Face Recognition",
        work_date_override: Optional[date_type] = None,
        tz_offset_hours: int = 7,
    ) -> AttendanceRecord:
        """
        Ghi nhận Check-in cho nhân viên trong ca/ngày làm việc.
        Trạng thái bản ghi đặt thành IN_PROGRESS (đã Check-in, chưa Check-out).
        """
        check_in = _ensure_utc(check_in)
        tz = timezone(timedelta(hours=tz_offset_hours))
        work_date = work_date_override or check_in.astimezone(tz).date()

        record = self._get_by_emp_date(employee_id, work_date)

        if record is None:
            record = AttendanceRecord(
                employee_id=employee_id,
                shift_id=shift_id,
                work_date=work_date,
                check_in=check_in,
                check_in_camera_id=camera_id,
                check_in_confidence=confidence,
                attendance_status="IN_PROGRESS",
                punctuality_status=punctuality_status,
                late_minutes=late_minutes or 0,
                early_leave_minutes=0,
                standard_work_minutes=standard_work_minutes,
                missing_minutes=0,
                extra_minutes=0,
                overtime_minutes=0,
                overtime_approved=False,
                needs_review=False,
                shift_start=shift_start,
                shift_end=shift_end,
                device_id=device_id,
                verify_method=verify_method,
                verify_status="Success",
                result="Accepted",
            )
            self._session.add(record)
        else:
            # Nếu đã có bản ghi (vd: chưa Check-in), chỉ cập nhật nếu chưa có check_in
            if record.check_in is None:
                record.check_in = check_in
                record.check_in_camera_id = camera_id
                record.check_in_confidence = confidence
                record.punctuality_status = punctuality_status
                record.late_minutes = late_minutes or 0
                if device_id:
                    record.device_id = device_id
                if record.check_out is not None:
                    record.attendance_status = "COMPLETED"
                    record.needs_review = False
                    record.review_reason = None

        self._session.flush()
        logger.info(
            "AttendanceRecord check-in: emp=%d date=%s status=%s punctuality=%s late=%s",
            employee_id, work_date, record.attendance_status, punctuality_status, late_minutes,
        )
        return record

    def upsert_check_out(
        self,
        employee_id: int,
        check_out: datetime,
        camera_id: Optional[int] = None,
        confidence: Optional[float] = None,
        early_leave_minutes: Optional[int] = None,
        break_minutes: Optional[int] = None,
        standard_work_minutes: Optional[int] = None,
        work_date_override: Optional[date_type] = None,
        tz_offset_hours: int = 7,
    ) -> Optional[AttendanceRecord]:
        """
        Ghi nhận Check-out cho nhân viên trong ca/ngày làm việc.
        - Nếu đã có Check-in: chuyển trạng thái sang COMPLETED và tính đầy đủ chỉ số thời gian.
        - Nếu chưa có Check-in: trả về None (hoặc dùng record_missing_checkin_checkout nếu muốn ghi ngoại lệ).
        """
        check_out = _ensure_utc(check_out)
        tz = timezone(timedelta(hours=tz_offset_hours))
        work_date = work_date_override or check_out.astimezone(tz).date()

        record = self._get_by_emp_date(employee_id, work_date)
        if record is None:
            logger.warning(
                "Không tìm thấy attendance_record check-in cho emp=%d date=%s.",
                employee_id, work_date,
            )
            return None

        existing_co = _ensure_utc(record.check_out) if record.check_out is not None else None
        if existing_co is None or check_out > existing_co:
            record.check_out = check_out
            record.check_out_camera_id = camera_id
            record.check_out_confidence = confidence

        if standard_work_minutes is not None:
            record.standard_work_minutes = standard_work_minutes

        if record.check_in is not None and record.check_out is not None:
            ci_utc = _ensure_utc(record.check_in)
            co_utc = _ensure_utc(record.check_out)
            presence_mins = max(0, int((co_utc - ci_utc).total_seconds() // 60))
            brk_mins = break_minutes if break_minutes is not None else (record.break_minutes or 0)
            brk_mins = min(brk_mins, presence_mins)
            actual_mins = max(0, presence_mins - brk_mins)
            std_mins = record.standard_work_minutes or 480

            record.total_presence_minutes = presence_mins
            record.break_minutes = brk_mins
            record.actual_work_minutes = actual_mins
            record.work_minutes = actual_mins
            record.missing_minutes = max(0, std_mins - actual_mins)
            record.extra_minutes = max(0, actual_mins - std_mins)
            record.attendance_status = "COMPLETED"

        if early_leave_minutes is not None:
            record.early_leave_minutes = early_leave_minutes
            if early_leave_minutes > 0:
                if record.punctuality_status == "LATE":
                    record.punctuality_status = "LATE_AND_EARLY"
                elif record.punctuality_status in ("ON_TIME", "UNKNOWN"):
                    record.punctuality_status = "EARLY_LEAVE"

        self._session.flush()
        logger.info(
            "AttendanceRecord check-out: emp=%d date=%s status=%s actual=%s missing=%s extra=%s",
            employee_id, work_date, record.attendance_status,
            record.actual_work_minutes, record.missing_minutes, record.extra_minutes,
        )
        return record

    def record_missing_checkin_checkout(
        self,
        employee_id: int,
        check_out: datetime,
        camera_id: Optional[int] = None,
        confidence: Optional[float] = None,
        shift_id: Optional[int] = None,
        shift_start: Optional[datetime] = None,
        shift_end: Optional[datetime] = None,
        standard_work_minutes: int = 480,
        device_id: Optional[str] = None,
        work_date_override: Optional[date_type] = None,
        tz_offset_hours: int = 7,
    ) -> AttendanceRecord:
        """
        Ngoại lệ: Nhân viên không Check-in nhưng có Check-out.
        Ghi nhận CheckOutTime với trạng thái INCOMPLETE và đánh dấu chờ HR đối soát.
        """
        check_out = _ensure_utc(check_out)
        tz = timezone(timedelta(hours=tz_offset_hours))
        work_date = work_date_override or check_out.astimezone(tz).date()

        record = self._get_by_emp_date(employee_id, work_date)
        if record is None:
            record = AttendanceRecord(
                employee_id=employee_id,
                shift_id=shift_id,
                work_date=work_date,
                check_in=None,
                check_out=check_out,
                check_out_camera_id=camera_id,
                check_out_confidence=confidence,
                attendance_status="INCOMPLETE",
                punctuality_status="UNKNOWN",
                standard_work_minutes=standard_work_minutes,
                needs_review=True,
                review_reason="MISSING_CHECK_IN",
                shift_start=shift_start,
                shift_end=shift_end,
                device_id=device_id,
                verify_method="Face Recognition",
                verify_status="Success",
                result="Accepted",
            )
            self._session.add(record)
        else:
            record.check_out = check_out
            record.check_out_camera_id = camera_id
            record.check_out_confidence = confidence
            if record.check_in is None:
                record.attendance_status = "INCOMPLETE"
                record.needs_review = True
                record.review_reason = "MISSING_CHECK_IN"

        self._session.flush()
        return record

    # ------------------------------------------------------------------
    # Đối soát cuối ca & Phê duyệt tăng ca
    # ------------------------------------------------------------------

    def reconcile_incomplete_records(
        self,
        work_date: date_type,
        cutoff_utc: Optional[datetime] = None,
    ) -> int:
        """
        Đối soát cuối ca: chuyển các bản ghi IN_PROGRESS (đã Check-in nhưng
        chưa Check-out sau khi ca kết thúc) sang INCOMPLETE + needs_review=True.
        Không tự động kết luận nghỉ không phép.
        """
        now = _ensure_utc(cutoff_utc) if cutoff_utc else datetime.now(timezone.utc)
        stmt = select(AttendanceRecord).where(
            and_(
                AttendanceRecord.work_date == work_date,
                AttendanceRecord.attendance_status == "IN_PROGRESS",
            )
        )
        records = list(self._session.execute(stmt).scalars().all())
        updated = 0
        for rec in records:
            # Chỉ chuyển sang INCOMPLETE nếu ca đã kết thúc (hoặc không có shift_end)
            if rec.shift_end is not None and _ensure_utc(rec.shift_end) > now:
                continue
            rec.attendance_status = "INCOMPLETE"
            rec.needs_review = True
            rec.review_reason = "MISSING_CHECK_OUT"
            updated += 1
        if updated > 0:
            self._session.flush()
            logger.info("Đối soát ngày %s: %d bản ghi → INCOMPLETE.", work_date, updated)
        return updated

    def approve_overtime(
        self,
        record_id: int,
        approved_minutes: int,
        note: Optional[str] = None,
    ) -> AttendanceRecord:
        """
        Phê duyệt tăng ca hợp lệ (OvertimeMinutes).
        Phân biệt rõ ExtraMinutes (tự động) với OvertimeMinutes (được duyệt).
        """
        if approved_minutes < 0:
            raise ValueError("approved_minutes phải >= 0")
        record = self.get_by_id(record_id)
        if record is None:
            raise ValueError(f"Không tìm thấy attendance_record id={record_id}")
        record.overtime_minutes = approved_minutes
        record.overtime_approved = approved_minutes > 0
        if note is not None:
            record.note = note
        record.record_source = "corrected"
        self._session.flush()
        return record

    # ------------------------------------------------------------------
    # Read — Phân cấp theo năm / tháng / ngày
    # ------------------------------------------------------------------

    def _get_by_emp_date(
        self, employee_id: int, work_date: date_type
    ) -> Optional[AttendanceRecord]:
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
        stmt = (
            select(AttendanceRecord)
            .where(AttendanceRecord.work_date == work_date)
            .order_by(AttendanceRecord.check_in)
        )
        return list(self._session.execute(stmt).scalars().all())

    def list_by_month(self, year: int, month: int) -> list[AttendanceRecord]:
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
        stmt = (
            select(AttendanceRecord)
            .where(extract("year", AttendanceRecord.work_date) == year)
            .order_by(AttendanceRecord.work_date, AttendanceRecord.employee_id)
        )
        return list(self._session.execute(stmt).scalars().all())

    def list_needs_review(self) -> list[AttendanceRecord]:
        """Lấy các bản ghi bất thường đang chờ HR đối soát."""
        stmt = (
            select(AttendanceRecord)
            .where(AttendanceRecord.needs_review.is_(True))
            .order_by(AttendanceRecord.work_date.desc())
        )
        return list(self._session.execute(stmt).scalars().all())

    def list_late_records(
        self,
        date_from: Optional[date_type] = None,
        date_to: Optional[date_type] = None,
        employee_id: Optional[int] = None,
    ) -> list[AttendanceRecord]:
        stmt = (
            select(AttendanceRecord)
            .where(AttendanceRecord.punctuality_status.in_(["LATE", "LATE_AND_EARLY"]))
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
        stmt = select(func.count()).where(
            and_(
                AttendanceRecord.employee_id == employee_id,
                AttendanceRecord.punctuality_status.in_(["LATE", "LATE_AND_EARLY"]),
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
        stmt = select(func.count()).where(
            and_(
                AttendanceRecord.employee_id == employee_id,
                AttendanceRecord.attendance_status.in_(["IN_PROGRESS", "COMPLETED"]),
            )
        )
        if year:
            stmt = stmt.where(extract("year", AttendanceRecord.work_date) == year)
        if month:
            stmt = stmt.where(extract("month", AttendanceRecord.work_date) == month)
        return self._session.execute(stmt).scalar_one() or 0

    # ------------------------------------------------------------------
    # Update (admin / HR đối soát thủ công)
    # ------------------------------------------------------------------

    def update_status(
        self,
        record_id: int,
        attendance_status: Optional[str] = None,
        punctuality_status: Optional[str] = None,
        note: Optional[str] = None,
        needs_review: Optional[bool] = None,
        record_source: str = "corrected",
    ) -> AttendanceRecord:
        record = self.get_by_id(record_id)
        if record is None:
            raise ValueError(f"Không tìm thấy attendance_record id={record_id}")

        valid_att = {"IN_PROGRESS", "COMPLETED", "INCOMPLETE"}
        valid_punc = {"ON_TIME", "LATE", "EARLY_LEAVE", "LATE_AND_EARLY", "UNKNOWN"}

        if attendance_status is not None:
            if attendance_status not in valid_att:
                raise ValueError(f"attendance_status không hợp lệ: {attendance_status!r}")
            record.attendance_status = attendance_status

        if punctuality_status is not None:
            if punctuality_status not in valid_punc:
                raise ValueError(f"punctuality_status không hợp lệ: {punctuality_status!r}")
            record.punctuality_status = punctuality_status

        if needs_review is not None:
            record.needs_review = needs_review
            if not needs_review:
                record.review_reason = None

        if note is not None:
            record.note = note

        record.record_source = record_source
        self._session.flush()
        logger.info(
            "AttendanceRecord updated: id=%d att=%s punc=%s",
            record_id, record.attendance_status, record.punctuality_status,
        )
        return record
