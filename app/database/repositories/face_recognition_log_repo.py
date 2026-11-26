"""
app/database/repositories/face_recognition_log_repo.py
Data-access layer cho bảng face_recognition_logs — Giai đoạn 10.

Chính sách lưu trữ:
  - Log SUCCESS  : giữ 30–90 ngày (cấu hình LOG_SUCCESS_RETENTION_DAYS)
  - Log FAILURE  : giữ 7–30 ngày  (cấu hình LOG_FAILURE_RETENTION_DAYS)
  - Ảnh sự kiện : giữ 7–30 ngày nếu thực sự cần và có chính sách bảo vệ
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, and_, delete
from sqlalchemy.orm import Session

from app.database.models import FaceRecognitionLog

logger = logging.getLogger(__name__)

# Thời hạn lưu trữ mặc định (ngày)
LOG_SUCCESS_RETENTION_DAYS: int = 60   # 30–90 ngày
LOG_FAILURE_RETENTION_DAYS: int = 14   # 7–30 ngày


class FaceRecognitionLogRepository:
    """CRUD + cleanup operations cho bảng face_recognition_logs."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def record_recognition(
        self,
        timestamp: datetime,
        result: str,
        employee_id: Optional[int] = None,
        camera_id: Optional[int] = None,
        confidence: Optional[float] = None,
        failure_reason: Optional[str] = None,
        event_snapshot_path: Optional[str] = None,
        retention_days: Optional[int] = None,
    ) -> FaceRecognitionLog:
        """
        Ghi một lần nhận diện vào log kỹ thuật.

        Tham số:
            timestamp            : Thời điểm nhận diện (UTC, timezone-aware)
            result               : 'SUCCESS' | 'FAILURE' | 'UNCERTAIN'
            employee_id          : ID nhân viên (None nếu FAILURE/UNCERTAIN)
            camera_id            : ID camera
            confidence           : Cosine similarity / confidence score (0.0–1.0)
            failure_reason       : Lý do thất bại (LOW_QUALITY, UNKNOWN_FACE, v.v.)
            event_snapshot_path  : Đường dẫn ảnh chụp sự kiện (nếu bật chụp)
            retention_days       : Override thời hạn lưu trữ (ngày)
        """
        valid_results = {"SUCCESS", "FAILURE", "UNCERTAIN"}
        if result not in valid_results:
            raise ValueError(f"result không hợp lệ: {result!r}")

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Tính thời hạn lưu trữ dựa vào kết quả
        if retention_days is None:
            retention_days = (
                LOG_SUCCESS_RETENTION_DAYS
                if result == "SUCCESS"
                else LOG_FAILURE_RETENTION_DAYS
            )
        expires_at = timestamp + timedelta(days=retention_days)

        log_entry = FaceRecognitionLog(
            employee_id=employee_id,
            camera_id=camera_id,
            timestamp=timestamp,
            confidence=confidence,
            result=result,
            failure_reason=failure_reason,
            event_snapshot_path=event_snapshot_path,
            expires_at=expires_at,
        )
        self._session.add(log_entry)
        self._session.flush()
        logger.debug(
            "FaceRecognitionLog: id=%d result=%s emp=%s conf=%s",
            log_entry.id, result, employee_id, confidence,
        )
        return log_entry

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, log_id: int) -> Optional[FaceRecognitionLog]:
        return self._session.get(FaceRecognitionLog, log_id)

    def list_by_employee(
        self,
        employee_id: int,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        result_filter: Optional[str] = None,
        limit: int = 100,
    ) -> list[FaceRecognitionLog]:
        """Lấy log nhận diện của một nhân viên theo bộ lọc."""
        stmt = (
            select(FaceRecognitionLog)
            .where(FaceRecognitionLog.employee_id == employee_id)
            .order_by(FaceRecognitionLog.timestamp.desc())
            .limit(limit)
        )
        if date_from:
            stmt = stmt.where(FaceRecognitionLog.timestamp >= date_from)
        if date_to:
            stmt = stmt.where(FaceRecognitionLog.timestamp <= date_to)
        if result_filter:
            stmt = stmt.where(FaceRecognitionLog.result == result_filter)
        return list(self._session.execute(stmt).scalars().all())

    def list_by_date(
        self,
        target_date: "date_type",
        tz_offset_hours: int = 7,
    ) -> list[FaceRecognitionLog]:
        """Lấy tất cả log trong một ngày (theo múi giờ local)."""
        from datetime import date as date_type

        tz = timezone(timedelta(hours=tz_offset_hours))
        day_start_local = datetime(
            target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=tz
        )
        day_end_local = datetime(
            target_date.year, target_date.month, target_date.day, 23, 59, 59, 999999, tzinfo=tz
        )
        day_start_utc = day_start_local.astimezone(timezone.utc)
        day_end_utc = day_end_local.astimezone(timezone.utc)

        stmt = (
            select(FaceRecognitionLog)
            .where(
                and_(
                    FaceRecognitionLog.timestamp >= day_start_utc,
                    FaceRecognitionLog.timestamp <= day_end_utc,
                )
            )
            .order_by(FaceRecognitionLog.timestamp)
        )
        return list(self._session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------
    # Cleanup (theo chính sách lưu trữ)
    # ------------------------------------------------------------------

    def delete_expired(self) -> int:
        """
        Xóa các log đã hết hạn lưu trữ (expires_at < now).
        Trả về số bản ghi đã xóa.
        Nên chạy định kỳ (scheduler hàng ngày).
        """
        now = datetime.now(timezone.utc)
        stmt = (
            delete(FaceRecognitionLog)
            .where(
                and_(
                    FaceRecognitionLog.expires_at.is_not(None),
                    FaceRecognitionLog.expires_at < now,
                )
            )
            .execution_options(synchronize_session=False)
        )
        result = self._session.execute(stmt)
        self._session.expire_all()
        count = result.rowcount
        if count > 0:
            logger.info("FaceRecognitionLog: đã xóa %d bản ghi hết hạn.", count)
        return count

    def delete_expired_snapshots(self, delete_file: bool = True) -> int:
        """
        Xóa đường dẫn ảnh sự kiện (và tùy chọn file thực) cho log đã hết hạn.
        Chính sách: ảnh giữ 7–30 ngày.
        """
        now = datetime.now(timezone.utc)
        stmt = select(FaceRecognitionLog).where(
            and_(
                FaceRecognitionLog.event_snapshot_path.is_not(None),
                FaceRecognitionLog.expires_at.is_not(None),
                FaceRecognitionLog.expires_at < now,
            )
        )
        logs = list(self._session.execute(stmt).scalars().all())
        count = 0
        for log in logs:
            if delete_file and log.event_snapshot_path:
                try:
                    import os
                    if os.path.exists(log.event_snapshot_path):
                        os.remove(log.event_snapshot_path)
                except OSError as exc:
                    logger.warning("Không thể xóa ảnh %s: %s", log.event_snapshot_path, exc)
            log.event_snapshot_path = None
            count += 1
        if count > 0:
            self._session.flush()
            logger.info("FaceRecognitionLog: đã xóa đường dẫn ảnh của %d bản ghi.", count)
        return count
