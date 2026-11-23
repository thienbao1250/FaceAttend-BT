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
