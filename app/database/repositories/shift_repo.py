"""
app/database/repositories/shift_repo.py
Data-access layer cho bảng shifts — Giai đoạn 11.
"""

from __future__ import annotations

import logging
from datetime import time
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Shift

logger = logging.getLogger(__name__)


class ShiftRepository:
    """CRUD operations cho bảng shifts."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        shift_code: str,
        name: str,
        start_time: time,
        end_time: time,
        break_start: Optional[time] = None,
        break_end: Optional[time] = None,
        break_minutes: int = 60,
        standard_work_minutes: int = 480,
        grace_period_minutes: int = 5,
        early_check_in_minutes: int = 120,
        late_check_out_minutes: int = 240,
        is_overnight: bool = False,
    ) -> Shift:
        """Tạo ca làm việc mới. Raise ValueError nếu shift_code đã tồn tại."""
        if self.get_by_code(shift_code):
            raise ValueError(f"shift_code đã tồn tại: {shift_code!r}")

        shift = Shift(
            shift_code=shift_code,
            name=name,
            start_time=start_time,
            end_time=end_time,
            break_start=break_start,
            break_end=break_end,
            break_minutes=break_minutes,
            standard_work_minutes=standard_work_minutes,
            grace_period_minutes=grace_period_minutes,
            early_check_in_minutes=early_check_in_minutes,
            late_check_out_minutes=late_check_out_minutes,
            is_overnight=is_overnight,
            is_active=True,
        )
        self._session.add(shift)
        self._session.flush()
        logger.info("Tạo ca làm việc: %s (%s)", shift_code, name)
        return shift

    def get_by_id(self, shift_id: int) -> Optional[Shift]:
        return self._session.get(Shift, shift_id)

    def get_by_code(self, shift_code: str) -> Optional[Shift]:
        stmt = select(Shift).where(Shift.shift_code == shift_code)
        return self._session.execute(stmt).scalar_one_or_none()

    def list_active(self) -> list[Shift]:
        stmt = select(Shift).where(Shift.is_active.is_(True)).order_by(Shift.id)
        return list(self._session.execute(stmt).scalars().all())

    def get_or_create_default(self) -> Shift:
        """Lấy ca hành chính mặc định 08:00–17:00 (nghỉ trưa 12:00–13:00), tạo nếu chưa có."""
        existing = self.get_by_code("HC_08_17")
        if existing is not None:
            return existing
        return self.create(
            shift_code="HC_08_17",
            name="Ca hành chính (08:00–17:00)",
            start_time=time(8, 0),
            end_time=time(17, 0),
            break_start=time(12, 0),
            break_end=time(13, 0),
            break_minutes=60,
            standard_work_minutes=480,
            grace_period_minutes=5,
            early_check_in_minutes=120,
            late_check_out_minutes=240,
            is_overnight=False,
        )
