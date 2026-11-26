"""
tests/unit/test_database.py
Unit tests cho Giai đoạn 9 — Database layer.

Dùng SQLite in-memory để test độc lập, không cần PostgreSQL.
Chạy: pytest tests/unit/test_database.py -v
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Đảm bảo import được app.*
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Fixtures: SQLite in-memory engine + session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def db_session() -> Session:
    """
    Tạo SQLite in-memory DB riêng cho mỗi test.
    Bảng được tạo mới → test hoàn toàn độc lập.
    """
    # Import sau khi sys.path đã được thiết lập
    from app.database.connection import Base
    import app.database.models  # noqa: F401

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


# ---------------------------------------------------------------------------
# Fixtures: numpy embedding giả
# ---------------------------------------------------------------------------

def _make_embedding(seed: int = 0) -> np.ndarray:
    """Tạo embedding giả (512,) float32 đã L2-normalize."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


# ===========================================================================
# Tests: embedding_to_bytes / bytes_to_embedding
# ===========================================================================

class TestEmbeddingConversion:

    def test_roundtrip_preserves_values(self):
        from app.database.models import embedding_to_bytes, bytes_to_embedding
        emb = _make_embedding(42)
        data = embedding_to_bytes(emb)
        recovered = bytes_to_embedding(data)
        np.testing.assert_array_almost_equal(emb, recovered, decimal=6)

    def test_bytes_length_correct(self):
        from app.database.models import embedding_to_bytes, EMBEDDING_DIM
        emb = _make_embedding(1)
        data = embedding_to_bytes(emb)
        assert len(data) == EMBEDDING_DIM * 4  # float32 = 4 bytes

    def test_wrong_shape_raises(self):
        from app.database.models import embedding_to_bytes
        bad_emb = np.zeros(256, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            embedding_to_bytes(bad_emb)

    def test_float64_converted_automatically(self):
        from app.database.models import embedding_to_bytes, bytes_to_embedding
        emb_f64 = np.ones(512, dtype=np.float64)
        emb_f64 /= np.linalg.norm(emb_f64)
        data = embedding_to_bytes(emb_f64)
        recovered = bytes_to_embedding(data)
        assert recovered.dtype == np.float32


# ===========================================================================
# Tests: EmployeeRepository
# ===========================================================================

class TestEmployeeRepository:

    def _repo(self, session):
        from app.database.repositories.employee_repo import EmployeeRepository
        return EmployeeRepository(session)

    def test_create_employee(self, db_session):
        repo = self._repo(db_session)
        emp = repo.create("NV001", "Nguyễn Văn A", department="IT")
        db_session.commit()
        assert emp.id is not None
        assert emp.employee_code == "NV001"
        assert emp.full_name == "Nguyễn Văn A"
        assert emp.status == "active"

    def test_get_by_code(self, db_session):
        repo = self._repo(db_session)
        repo.create("NV002", "Trần Thị B")
        db_session.commit()
        found = repo.get_by_code("NV002")
        assert found is not None
        assert found.full_name == "Trần Thị B"

    def test_get_by_code_not_found(self, db_session):
        repo = self._repo(db_session)
        assert repo.get_by_code("NONEXIST") is None

    def test_duplicate_code_raises(self, db_session):
        repo = self._repo(db_session)
        repo.create("NV003", "Lê Văn C")
        db_session.commit()
        with pytest.raises(ValueError, match="employee_code đã tồn tại"):
            repo.create("NV003", "Người Khác")

    def test_duplicate_email_raises(self, db_session):
        repo = self._repo(db_session)
        repo.create("NV004", "A", email="a@test.com")
        db_session.commit()
        with pytest.raises(ValueError, match="email đã tồn tại"):
            repo.create("NV005", "B", email="a@test.com")

    def test_list_active_excludes_inactive(self, db_session):
        repo = self._repo(db_session)
        repo.create("NV010", "Active")
        emp2 = repo.create("NV011", "Inactive")
        db_session.commit()
        repo.set_status(emp2.id, "inactive")
        db_session.commit()
        active = repo.list_active()
        codes = [e.employee_code for e in active]
        assert "NV010" in codes
        assert "NV011" not in codes

    def test_set_status_deleted(self, db_session):
        repo = self._repo(db_session)
        emp = repo.create("NV020", "To Delete")
        db_session.commit()
        repo.set_status(emp.id, "deleted")
        db_session.commit()
        # list_all không bao gồm deleted theo mặc định
        all_employees = repo.list_all(include_deleted=False)
        assert all(e.employee_code != "NV020" for e in all_employees)
        # nhưng include_deleted=True thì có
        all_with_deleted = repo.list_all(include_deleted=True)
        assert any(e.employee_code == "NV020" for e in all_with_deleted)

    def test_invalid_status_raises(self, db_session):
        repo = self._repo(db_session)
        emp = repo.create("NV030", "Test")
        db_session.commit()
        with pytest.raises(ValueError, match="status không hợp lệ"):
            repo.set_status(emp.id, "banned")

    def test_update_employee(self, db_session):
        repo = self._repo(db_session)
        emp = repo.create("NV040", "Old Name", department="Finance")
        db_session.commit()
        repo.update(emp.id, full_name="New Name", department="HR")
        db_session.commit()
        updated = repo.get_by_id(emp.id)
        assert updated.full_name == "New Name"
        assert updated.department == "HR"

    def test_hard_delete(self, db_session):
        repo = self._repo(db_session)
        emp = repo.create("NV050", "Delete Me")
        db_session.commit()
        result = repo.hard_delete(emp.id)
        db_session.commit()
        assert result is True
        assert repo.get_by_id(emp.id) is None

    def test_hard_delete_nonexistent(self, db_session):
        repo = self._repo(db_session)
        assert repo.hard_delete(99999) is False


# ===========================================================================
# Tests: FaceTemplateRepository
# ===========================================================================

class TestFaceTemplateRepository:

    def _emp(self, session, code="EMP001", name="Test Employee"):
        from app.database.repositories.employee_repo import EmployeeRepository
        repo = EmployeeRepository(session)
        emp = repo.create(code, name)
        session.commit()
        return emp

    def _repo(self, session):
        from app.database.repositories.face_template_repo import FaceTemplateRepository
        return FaceTemplateRepository(session)

    def test_add_template(self, db_session):
        emp = self._emp(db_session)
        repo = self._repo(db_session)
        emb = _make_embedding(10)
        tpl = repo.add(emp.id, emb, model_name="w600k_r50", quality_score=0.95)
        db_session.commit()
        assert tpl.id is not None
        assert tpl.employee_id == emp.id
        assert tpl.is_active is True

    def test_embedding_roundtrip_via_db(self, db_session):
        emp = self._emp(db_session, "EMP002")
        repo = self._repo(db_session)
        emb = _make_embedding(20)
        tpl = repo.add(emp.id, emb, model_name="w600k_r50")
        db_session.commit()
        # Đọc lại từ DB
        tpl_loaded = repo.get_by_id(tpl.id)
        recovered = tpl_loaded.get_embedding()
        np.testing.assert_array_almost_equal(emb, recovered, decimal=6)

    def test_invalid_embedding_shape_raises(self, db_session):
        emp = self._emp(db_session, "EMP003")
        repo = self._repo(db_session)
        bad = np.zeros(256, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            repo.add(emp.id, bad, model_name="w600k_r50")

    def test_nan_embedding_raises(self, db_session):
        emp = self._emp(db_session, "EMP004")
        repo = self._repo(db_session)
        bad = np.full(512, float("nan"), dtype=np.float32)
        with pytest.raises(ValueError, match="NaN hoặc Inf"):
            repo.add(emp.id, bad, model_name="w600k_r50")

    def test_deactivate_template(self, db_session):
        emp = self._emp(db_session, "EMP005")
        repo = self._repo(db_session)
        tpl = repo.add(emp.id, _make_embedding(30), model_name="w600k_r50")
        db_session.commit()
        result = repo.deactivate(tpl.id)
        db_session.commit()
        assert result is True
        assert repo.get_by_id(tpl.id).is_active is False

    def test_list_active_excludes_inactive(self, db_session):
        emp = self._emp(db_session, "EMP006")
        repo = self._repo(db_session)
        t1 = repo.add(emp.id, _make_embedding(31), model_name="w600k_r50")
        t2 = repo.add(emp.id, _make_embedding(32), model_name="w600k_r50")
        db_session.commit()
        repo.deactivate(t2.id)
        db_session.commit()
        active = repo.list_active_by_employee(emp.id, model_name="w600k_r50")
        ids = [t.id for t in active]
        assert t1.id in ids
        assert t2.id not in ids

    def test_load_gallery(self, db_session):
        emp1 = self._emp(db_session, "EMP007", "Person A")
        emp2 = self._emp(db_session, "EMP008", "Person B")
        repo = self._repo(db_session)
        repo.add(emp1.id, _make_embedding(40), model_name="w600k_r50")
        repo.add(emp1.id, _make_embedding(41), model_name="w600k_r50")
        repo.add(emp2.id, _make_embedding(42), model_name="w600k_r50")
        db_session.commit()
        gallery = repo.load_gallery("w600k_r50", "1.0")
        assert emp1.id in gallery
        assert emp2.id in gallery
        assert len(gallery[emp1.id]) == 2
        assert len(gallery[emp2.id]) == 1

    def test_delete_by_employee(self, db_session):
        emp = self._emp(db_session, "EMP009")
        repo = self._repo(db_session)
        repo.add(emp.id, _make_embedding(50), model_name="w600k_r50")
        repo.add(emp.id, _make_embedding(51), model_name="w600k_r50")
        db_session.commit()
        count = repo.delete_by_employee(emp.id)
        db_session.commit()
        assert count == 2
        assert repo.count_by_employee(emp.id, active_only=False) == 0

    def test_count_by_employee(self, db_session):
        emp = self._emp(db_session, "EMP010")
        repo = self._repo(db_session)
        repo.add(emp.id, _make_embedding(60), model_name="w600k_r50")
        repo.add(emp.id, _make_embedding(61), model_name="w600k_r50")
        db_session.commit()
        assert repo.count_by_employee(emp.id) == 2


# ===========================================================================
# Tests: AttendanceRepository
# ===========================================================================

class TestAttendanceRepository:

    def _emp(self, session, code="ATT001"):
        from app.database.repositories.employee_repo import EmployeeRepository
        repo = EmployeeRepository(session)
        emp = repo.create(code, f"Nhân viên {code}")
        session.commit()
        return emp

    def _repo(self, session):
        from app.database.repositories.attendance_repo import AttendanceRepository
        return AttendanceRepository(session)

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def test_record_check_in(self, db_session):
        emp = self._emp(db_session)
        repo = self._repo(db_session)
        event = repo.record_event(
            employee_id=emp.id,
            event_type="check_in",
            event_time=self._now_utc(),
            confidence=0.92,
        )
        db_session.commit()
        assert event.id is not None
        assert event.event_type == "check_in"
        assert event.verify_status == "auto"

    def test_record_check_out(self, db_session):
        emp = self._emp(db_session, "ATT002")
        repo = self._repo(db_session)
        t_in = self._now_utc()
        t_out = t_in + timedelta(hours=8)
        repo.record_event(emp.id, "check_in", t_in, confidence=0.90)
        event_out = repo.record_event(emp.id, "check_out", t_out, confidence=0.88)
        db_session.commit()
        assert event_out.event_type == "check_out"

    def test_invalid_event_type_raises(self, db_session):
        emp = self._emp(db_session, "ATT003")
        repo = self._repo(db_session)
        with pytest.raises(ValueError, match="event_type không hợp lệ"):
            repo.record_event(emp.id, "work", self._now_utc())

    def test_naive_datetime_accepted(self, db_session):
        """
        Datetime không có timezone được chấp nhận và tự gắn UTC bởi repo.
        SQLite không lưu tzinfo → kiểm tra giá trị thời gian vẫn đúng.
        """
        emp = self._emp(db_session, "ATT004")
        repo = self._repo(db_session)
        naive = datetime(2026, 1, 1, 8, 0, 0)  # no tzinfo
        event = repo.record_event(emp.id, "check_in", naive)
        db_session.commit()
        # Repo phải đã gắn UTC trước khi lưu (event_time.tzinfo=UTC trước commit)
        # SQLite strip tzinfo khi đọc lại — kiểm tra giá trị thời gian không đổi
        assert event.event_time.year == 2026
        assert event.event_time.month == 1
        assert event.event_time.day == 1
        assert event.event_time.hour == 8

    def test_get_last_event(self, db_session):
        emp = self._emp(db_session, "ATT005")
        repo = self._repo(db_session)
        t1 = self._now_utc() - timedelta(hours=2)
        t2 = self._now_utc()
        repo.record_event(emp.id, "check_in", t1)
        repo.record_event(emp.id, "check_out", t2)
        db_session.commit()
        last = repo.get_last_event(emp.id)
        assert last.event_type == "check_out"

    def test_list_by_employee(self, db_session):
        emp = self._emp(db_session, "ATT006")
        repo = self._repo(db_session)
        for i in range(3):
            t = self._now_utc() + timedelta(hours=i)
            repo.record_event(emp.id, "check_in", t)
        db_session.commit()
        events = repo.list_by_employee(emp.id)
        assert len(events) == 3

    def test_list_by_employee_date_filter(self, db_session):
        emp = self._emp(db_session, "ATT007")
        repo = self._repo(db_session)
        t_today = datetime.now(timezone.utc)
        t_old = t_today - timedelta(days=10)
        repo.record_event(emp.id, "check_in", t_today)
        repo.record_event(emp.id, "check_in", t_old)
        db_session.commit()
        # Chỉ lấy 5 ngày gần đây
        date_from = t_today - timedelta(days=5)
        events = repo.list_by_employee(emp.id, date_from=date_from)
        assert len(events) == 1

    def test_update_verify_status(self, db_session):
        emp = self._emp(db_session, "ATT008")
        repo = self._repo(db_session)
        event = repo.record_event(emp.id, "check_in", self._now_utc())
        db_session.commit()
        repo.update_verify_status(event.id, "disputed", note="Nhân viên khiếu nại")
        db_session.commit()
        reloaded = repo.get_by_id(event.id)
        assert reloaded.verify_status == "disputed"
        assert reloaded.note == "Nhân viên khiếu nại"

    def test_update_invalid_verify_status_raises(self, db_session):
        emp = self._emp(db_session, "ATT009")
        repo = self._repo(db_session)
        event = repo.record_event(emp.id, "check_in", self._now_utc())
        db_session.commit()
        with pytest.raises(ValueError, match="verify_status không hợp lệ"):
            repo.update_verify_status(event.id, "unknown_status")

    def test_list_by_date(self, db_session):
        from datetime import date as date_type
        emp = self._emp(db_session, "ATT010")
        repo = self._repo(db_session)
        # Tạo sự kiện hôm nay (UTC+7)
        today_utc = datetime.now(timezone.utc)
        repo.record_event(emp.id, "check_in", today_utc)
        db_session.commit()
        today_local = date_type.today()
        events = repo.list_by_date(today_local, tz_offset_hours=7)
        assert any(e.employee_id == emp.id for e in events)


# ===========================================================================
# Tests: FaceRecognitionLogRepository — Giai đoạn 10
# ===========================================================================

class TestFaceRecognitionLogRepository:

    def _emp(self, session, code="LOG_EMP_01"):
        from app.database.repositories.employee_repo import EmployeeRepository
        repo = EmployeeRepository(session)
        emp = repo.create(code, f"Nhân viên {code}")
        session.commit()
        return emp

    def _cam(self, session, code="CAM001"):
        from app.database.models import Camera
        cam = Camera(camera_code=code, name="Cổng chính", status="online")
        session.add(cam)
        session.commit()
        return cam

    def _repo(self, session):
        from app.database.repositories.face_recognition_log_repo import FaceRecognitionLogRepository
        return FaceRecognitionLogRepository(session)

    def test_record_success_log(self, db_session):
        emp = self._emp(db_session)
        cam = self._cam(db_session)
        repo = self._repo(db_session)
        now = datetime.now(timezone.utc)
        log = repo.record_recognition(
            timestamp=now,
            result="SUCCESS",
            employee_id=emp.id,
            camera_id=cam.id,
            confidence=0.95,
        )
        db_session.commit()
        assert log.id is not None
        assert log.result == "SUCCESS"
        assert log.failure_reason is None
        assert log.confidence == pytest.approx(0.95)
        assert log.expires_at is not None

    def test_record_failure_log_with_null_employee(self, db_session):
        cam = self._cam(db_session, "CAM002")
        repo = self._repo(db_session)
        now = datetime.now(timezone.utc)
        log = repo.record_recognition(
            timestamp=now,
            result="FAILURE",
            employee_id=None,
            camera_id=cam.id,
            confidence=0.12,
            failure_reason="UNKNOWN_FACE",
        )
        db_session.commit()
        assert log.id is not None
        assert log.employee_id is None
        assert log.result == "FAILURE"
        assert log.failure_reason == "UNKNOWN_FACE"

    def test_retention_policy_success_vs_failure(self, db_session):
        from app.database.repositories.face_recognition_log_repo import (
            LOG_SUCCESS_RETENTION_DAYS,
            LOG_FAILURE_RETENTION_DAYS,
        )
        emp = self._emp(db_session, "LOG_EMP_02")
        repo = self._repo(db_session)
        now = datetime(2026, 11, 26, 8, 0, 0, tzinfo=timezone.utc)
        log_ok = repo.record_recognition(timestamp=now, result="SUCCESS", employee_id=emp.id)
        log_fail = repo.record_recognition(timestamp=now, result="FAILURE", failure_reason="LOW_QUALITY")
        db_session.commit()

        assert (log_ok.expires_at.replace(tzinfo=timezone.utc) - now).days == LOG_SUCCESS_RETENTION_DAYS
        assert (log_fail.expires_at.replace(tzinfo=timezone.utc) - now).days == LOG_FAILURE_RETENTION_DAYS
        assert 30 <= LOG_SUCCESS_RETENTION_DAYS <= 90
        assert 7 <= LOG_FAILURE_RETENTION_DAYS <= 30

    def test_invalid_result_raises(self, db_session):
        repo = self._repo(db_session)
        with pytest.raises(ValueError, match="result không hợp lệ"):
            repo.record_recognition(timestamp=datetime.now(timezone.utc), result="UNKNOWN_STATUS")

    def test_delete_expired_removes_only_expired(self, db_session):
        emp = self._emp(db_session, "LOG_EMP_03")
        repo = self._repo(db_session)
        old_ts = datetime.now(timezone.utc) - timedelta(days=40)
        now_ts = datetime.now(timezone.utc)
        old_log = repo.record_recognition(
            timestamp=old_ts, result="FAILURE", failure_reason="UNKNOWN_FACE", retention_days=14
        )
        fresh_log = repo.record_recognition(
            timestamp=now_ts, result="SUCCESS", employee_id=emp.id, retention_days=60
        )
        db_session.commit()
        old_id, fresh_id = old_log.id, fresh_log.id

        deleted = repo.delete_expired()
        db_session.commit()
        assert deleted == 1
        assert repo.get_by_id(old_id) is None
        assert repo.get_by_id(fresh_id) is not None

    def test_list_by_employee_and_date(self, db_session):
        from datetime import date as date_type
        emp = self._emp(db_session, "LOG_EMP_04")
        repo = self._repo(db_session)
        ts = datetime(2026, 11, 26, 1, 0, 0, tzinfo=timezone.utc)  # 08:00:00 GMT+7
        repo.record_recognition(timestamp=ts, result="SUCCESS", employee_id=emp.id, confidence=0.91)
        db_session.commit()

        by_emp = repo.list_by_employee(emp.id)
        assert len(by_emp) == 1
        by_day = repo.list_by_date(date_type(2026, 11, 26), tz_offset_hours=7)
        assert any(item.employee_id == emp.id for item in by_day)


# ===========================================================================
# Tests: AttendanceRecordRepository — Giai đoạn 10
# ===========================================================================

class TestAttendanceRecordRepository:

    def _emp(self, session, code="FA-IT-2026-0001"):
        from app.database.repositories.employee_repo import EmployeeRepository
        repo = EmployeeRepository(session)
        emp = repo.create(code, f"Nhân viên {code}")
        session.commit()
        return emp

    def _repo(self, session):
        from app.database.repositories.attendance_record_repo import AttendanceRecordRepository
        return AttendanceRecordRepository(session)

    def test_upsert_check_in_creates_record(self, db_session):
        emp = self._emp(db_session)
        repo = self._repo(db_session)
        # 2026-11-26 08:05:30 GMT+7 = 01:05:30 UTC
        checkin_utc = datetime(2026, 11, 26, 1, 5, 30, tzinfo=timezone.utc)
        rec = repo.upsert_check_in(
            employee_id=emp.id,
            check_in=checkin_utc,
            confidence=0.94,
            punctuality_status="LATE",
            late_minutes=5,
            device_id="CAM001",
        )
        db_session.commit()

        assert rec.id is not None
        assert rec.work_date.year == 2026
        assert rec.work_date.month == 11
        assert rec.work_date.day == 26
        assert rec.attendance_status == "IN_PROGRESS"
        assert rec.punctuality_status == "LATE"
        assert rec.late_minutes == 5
        assert rec.verify_method == "Face Recognition"
        assert rec.verify_status == "Success"
        assert rec.result == "Accepted"

    def test_upsert_check_out_calculates_work_minutes(self, db_session):
        emp = self._emp(db_session, "FA-IT-2026-0002")
        repo = self._repo(db_session)
        checkin_utc = datetime(2026, 11, 26, 1, 0, 15, tzinfo=timezone.utc)   # 08:00:15 GMT+7
        checkout_utc = datetime(2026, 11, 26, 10, 5, 20, tzinfo=timezone.utc)  # 17:05:20 GMT+7

        repo.upsert_check_in(emp.id, checkin_utc, punctuality_status="ON_TIME", late_minutes=0)
        rec = repo.upsert_check_out(emp.id, checkout_utc, confidence=0.93, break_minutes=60)
        db_session.commit()

        assert rec is not None
        assert rec.check_out is not None
        assert rec.attendance_status == "COMPLETED"
        assert rec.total_presence_minutes == 545
        assert rec.break_minutes == 60
        assert rec.actual_work_minutes == 485
        assert rec.work_minutes == 485
        assert rec.extra_minutes == 5
        assert rec.overtime_minutes == 0

    def test_to_display_dict_matches_business_format(self, db_session):
        emp = self._emp(db_session, "FA-IT-2026-0003")
        repo = self._repo(db_session)
        checkin_utc = datetime(2026, 11, 26, 1, 5, 30, tzinfo=timezone.utc)  # 2026-11-26 08:05:30 GMT+7
        rec = repo.upsert_check_in(
            employee_id=emp.id,
            check_in=checkin_utc,
            device_id="CAM001",
            punctuality_status="ON_TIME",
            late_minutes=0,
        )
        db_session.commit()

        d = rec.to_display_dict(tz_offset_hours=7)
        assert d["Employee ID"] == "FA-IT-2026-0003"
        assert d["Thời gian"] == "2026-11-26 08:05:30"
        assert d["Device ID"] == "CAM001"
        assert d["Phương thức"] == "Face Recognition"
        assert d["Event Type"] == "Check-in"
        assert d["Verify Status"] == "Success"
        assert d["Result"] == "Accepted"
        assert d["Record ID"] == rec.id

    def test_query_by_year_month_day(self, db_session):
        from datetime import date as date_type
        emp = self._emp(db_session, "FA-IT-2026-0004")
        repo = self._repo(db_session)

        d1 = datetime(2026, 11, 26, 1, 0, 0, tzinfo=timezone.utc)
        d2 = datetime(2026, 11, 27, 1, 15, 0, tzinfo=timezone.utc)
        d3 = datetime(2026, 10, 5, 1, 0, 0, tzinfo=timezone.utc)

        repo.upsert_check_in(emp.id, d1, punctuality_status="ON_TIME", late_minutes=0)
        repo.upsert_check_in(emp.id, d2, punctuality_status="LATE", late_minutes=15)
        repo.upsert_check_in(emp.id, d3, punctuality_status="ON_TIME", late_minutes=0)
        db_session.commit()

        assert len(repo.list_by_date(date_type(2026, 11, 26))) == 1
        assert len(repo.list_by_month(2026, 11)) == 2
        assert len(repo.list_by_year(2026)) == 3
        assert repo.count_late_days(emp.id, year=2026, month=11) == 1
        assert repo.count_present_days(emp.id, year=2026) == 3

    def test_update_status(self, db_session):
        emp = self._emp(db_session, "FA-IT-2026-0005")
        repo = self._repo(db_session)
        rec = repo.upsert_check_in(
            emp.id, datetime(2026, 11, 26, 1, 20, 0, tzinfo=timezone.utc),
            punctuality_status="LATE", late_minutes=20,
        )
        db_session.commit()
        repo.update_status(rec.id, punctuality_status="ON_TIME", note="Xin phép đi muộn được duyệt")
        db_session.commit()
        updated = repo.get_by_id(rec.id)
        assert updated.punctuality_status == "ON_TIME"
        assert updated.record_source == "corrected"
        assert updated.note == "Xin phép đi muộn được duyệt"


    def test_missing_checkin_checkout_marks_incomplete_and_needs_review(self, db_session):
        emp = self._emp(db_session, "FA-IT-2026-0006")
        repo = self._repo(db_session)
        checkout_utc = datetime(2026, 11, 26, 10, 0, 0, tzinfo=timezone.utc)
        rec = repo.record_missing_checkin_checkout(emp.id, checkout_utc)
        db_session.commit()
        assert rec.attendance_status == "INCOMPLETE"
        assert rec.check_in is None
        assert rec.needs_review is True
        assert rec.review_reason == "MISSING_CHECK_IN"
        assert len(repo.list_needs_review()) >= 1

    def test_reconcile_incomplete_records_and_approve_overtime(self, db_session):
        from datetime import date as date_type
        emp = self._emp(db_session, "FA-IT-2026-0007")
        repo = self._repo(db_session)
        shift_end_utc = datetime(2026, 11, 26, 10, 0, 0, tzinfo=timezone.utc)
        rec = repo.upsert_check_in(
            emp.id,
            datetime(2026, 11, 26, 1, 0, 0, tzinfo=timezone.utc),
            shift_end=shift_end_utc,
        )
        db_session.commit()

        # Trước khi hết ca -> không chuyển INCOMPLETE
        assert repo.reconcile_incomplete_records(date_type(2026, 11, 26), shift_end_utc - timedelta(minutes=10)) == 0
        # Sau khi hết ca -> chuyển INCOMPLETE
        assert repo.reconcile_incomplete_records(date_type(2026, 11, 26), shift_end_utc + timedelta(minutes=10)) == 1
        db_session.commit()
        reloaded = repo.get_by_id(rec.id)
        assert reloaded.attendance_status == "INCOMPLETE"
        assert reloaded.needs_review is True
        assert reloaded.review_reason == "MISSING_CHECK_OUT"

        # Approve overtime
        repo.approve_overtime(rec.id, approved_minutes=60, note="Đã duyệt OT")
        db_session.commit()
        reloaded2 = repo.get_by_id(rec.id)
        assert reloaded2.overtime_minutes == 60
        assert reloaded2.overtime_approved is True


# ===========================================================================
# Tests: ShiftRepository — Giai đoạn 11
# ===========================================================================

class TestShiftRepository:

    def test_get_or_create_default_shift(self, db_session):
        from app.database.repositories.shift_repo import ShiftRepository
        repo = ShiftRepository(db_session)
        s1 = repo.get_or_create_default()
        db_session.commit()
        s2 = repo.get_or_create_default()
        assert s1.id == s2.id
        assert s1.shift_code == "HC_08_17"
        assert s1.standard_work_minutes == 480
        assert s1.break_minutes == 60
