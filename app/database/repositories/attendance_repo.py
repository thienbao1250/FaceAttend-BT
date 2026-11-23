"""
app/database/repositories/attendance_repo.py
Data-access layer cho bảng attendance_events — Giai đoạn 9.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, date
from typing import Optional

from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from app.database.models import AttendanceEvent

logger = logging.getLogger(__name__)


class AttendanceRepository:
    """CRUD operations cho bảng attendance_events."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def record_event(
        self,
        employee_id: int,
        event_type: str,
        event_time: datetime,
        camera_id: Optional[int] = None,
        confidence: Optional[float] = None,
        verify_status: str = "auto",
        note: Optional[str] = None,
    ) -> AttendanceEvent:
        """
        Ghi một sự kiện chấm công vào DB.

        event_type: 'check_in' | 'check_out'
        event_time: datetime UTC (timezone-aware)
        verify_status: 'auto' (nhận diện tự động) | 'manual' (admin chỉnh) | 'disputed' (khiếu nại)
        """
        valid_types = {"check_in", "check_out"}
        if event_type not in valid_types:
            raise ValueError(f"event_type không hợp lệ: {event_type!r}")

        # Đảm bảo event_time có timezone UTC
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)

        event = AttendanceEvent(
            employee_id=employee_id,
            camera_id=camera_id,
            event_type=event_type,
            event_time=event_time,
            confidence=confidence,
            verify_status=verify_status,
            note=note,
        )
        self._session.add(event)
        self._session.flush()
        logger.info(
            "Ghi sự kiện %s: employee_id=%d time=%s camera_id=%s confidence=%.3f",
            event_type, employee_id, event_time.isoformat(),
            camera_id, confidence or 0.0,
        )
        return event

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, event_id: int) -> Optional[AttendanceEvent]:
        return self._session.get(AttendanceEvent, event_id)

    def get_last_event(
        self,
        employee_id: int,
        camera_id: Optional[int] = None,
    ) -> Optional[AttendanceEvent]:
        """Lấy sự kiện gần nhất của nhân viên (có thể lọc theo camera)."""
        stmt = (
            select(AttendanceEvent)
            .where(AttendanceEvent.employee_id == employee_id)
            .order_by(AttendanceEvent.event_time.desc())
            .limit(1)
        )
        if camera_id is not None:
            stmt = stmt.where(AttendanceEvent.camera_id == camera_id)
        return self._session.execute(stmt).scalar_one_or_none()

    def list_by_employee(
        self,
        employee_id: int,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AttendanceEvent]:
        """
        Lấy lịch sử chấm công của một nhân viên theo khoảng thời gian.
        date_from / date_to: datetime UTC (timezone-aware).
        """
        stmt = (
            select(AttendanceEvent)
            .where(AttendanceEvent.employee_id == employee_id)
            .order_by(AttendanceEvent.event_time.desc())
            .limit(limit)
        )
        if date_from:
            stmt = stmt.where(AttendanceEvent.event_time >= date_from)
        if date_to:
            stmt = stmt.where(AttendanceEvent.event_time <= date_to)
        return list(self._session.execute(stmt).scalars().all())

    def list_by_date(
        self,
        target_date: date,
        tz_offset_hours: int = 7,
    ) -> list[AttendanceEvent]:
        """
        Lấy tất cả sự kiện trong một ngày (theo múi giờ tz_offset_hours).
        Chuyển ngày target_date sang khoảng UTC tương ứng.
        """
        from datetime import timedelta

        # Tính khoảng UTC của ngày đó theo múi giờ
        tz = timezone(timedelta(hours=tz_offset_hours))
        day_start_local = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=tz)
        day_end_local = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59, 999999, tzinfo=tz)
        day_start_utc = day_start_local.astimezone(timezone.utc)
        day_end_utc = day_end_local.astimezone(timezone.utc)

        stmt = (
            select(AttendanceEvent)
            .where(
                and_(
                    AttendanceEvent.event_time >= day_start_utc,
                    AttendanceEvent.event_time <= day_end_utc,
                )
            )
            .order_by(AttendanceEvent.event_time)
        )
        return list(self._session.execute(stmt).scalars().all())

    def count_events_today(self, employee_id: int, tz_offset_hours: int = 7) -> int:
        """Đếm số sự kiện hôm nay của một nhân viên."""
        from datetime import timedelta, date as date_type

        today = date_type.today()
        events = self.list_by_date(today, tz_offset_hours)
        return sum(1 for e in events if e.employee_id == employee_id)

    # ------------------------------------------------------------------
    # Update (admin chỉnh sửa thủ công)
    # ------------------------------------------------------------------

    def update_verify_status(
        self,
        event_id: int,
        verify_status: str,
        note: Optional[str] = None,
    ) -> AttendanceEvent:
        """Cập nhật trạng thái xác minh của sự kiện (manual/disputed)."""
        valid = {"auto", "manual", "disputed"}
        if verify_status not in valid:
            raise ValueError(f"verify_status không hợp lệ: {verify_status!r}")
        event = self.get_by_id(event_id)
        if event is None:
            raise ValueError(f"Không tìm thấy attendance_event id={event_id}")
        event.verify_status = verify_status
        if note is not None:
            event.note = note
        self._session.flush()
        logger.info(
            "Cập nhật verify_status sự kiện id=%d → %s", event_id, verify_status
        )
        return event
