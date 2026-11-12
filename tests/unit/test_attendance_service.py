"""
tests/unit/test_attendance_service.py
Unit tests cho AttendanceService — Giai đoạn 8.

Chạy: pytest tests/unit/test_attendance_service.py -v
"""

from __future__ import annotations

import time
from datetime import timezone, timedelta
from unittest.mock import patch

import pytest

from app.attendance.attendance_service import (
    AttendanceService,
    AttendanceEvent,
    AttendanceDecision,
    EventType,
    SkipReason,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_MIN_TOGGLE_SECONDS,
)
from app.vision.face_matcher import MatchResult, MatchStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _known(emp_id: str = "EMP001", sim: float = 0.55) -> MatchResult:
    return MatchResult(
        status=MatchStatus.KNOWN,
        employee_id=emp_id,
        similarity=sim,
        num_candidates=1,
    )


def _unknown(sim: float = 0.05) -> MatchResult:
    return MatchResult(
        status=MatchStatus.UNKNOWN,
        employee_id=None,
        similarity=sim,
        num_candidates=1,
    )


def _uncertain(emp_id: str = "EMP001", sim: float = 0.28) -> MatchResult:
    return MatchResult(
        status=MatchStatus.UNCERTAIN,
        employee_id=None,
        similarity=sim,
        num_candidates=1,
    )


T0 = 1_700_000_000.0   # Timestamp giả cố định dùng trong test


@pytest.fixture
def svc():
    return AttendanceService(
        cooldown_seconds=30.0,
        min_toggle_seconds=60.0,
    )


# ---------------------------------------------------------------------------
# Test: UNKNOWN / UNCERTAIN không tạo sự kiện
# ---------------------------------------------------------------------------

class TestNonKnownResults:
    def test_unknown_skipped(self, svc):
        d = svc.process(_unknown(), timestamp=T0)
        assert d.created is False
        assert d.skip_reason == SkipReason.UNKNOWN
        assert svc.total_events == 0

    def test_uncertain_skipped(self, svc):
        d = svc.process(_uncertain(), timestamp=T0)
        assert d.created is False
        assert d.skip_reason == SkipReason.UNCERTAIN
        assert svc.total_events == 0

    def test_unknown_message_informative(self, svc):
        d = svc.process(_unknown(), timestamp=T0)
        assert d.message  # không rỗng


# ---------------------------------------------------------------------------
# Test: CHECK_IN cơ bản
# ---------------------------------------------------------------------------

class TestCheckIn:
    def test_first_recognition_creates_check_in(self, svc):
        d = svc.process(_known(), timestamp=T0)
        assert d.created is True
        assert d.event is not None
        assert d.event.event_type == EventType.CHECK_IN

    def test_event_stores_correct_employee(self, svc):
        d = svc.process(_known("EMP001"), timestamp=T0)
        assert d.event.employee_id == "EMP001"

    def test_event_stores_timestamp(self, svc):
        d = svc.process(_known(), timestamp=T0)
        assert d.event.timestamp_utc == T0

    def test_event_stores_camera_id(self, svc):
        d = svc.process(_known(), camera_id="cam_lobby", timestamp=T0)
        assert d.event.camera_id == "cam_lobby"

    def test_event_stores_similarity(self, svc):
        d = svc.process(_known(sim=0.72), timestamp=T0)
        assert abs(d.event.similarity - 0.72) < 1e-5

    def test_total_events_increments(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)
        svc.process(_known("EMP002"), timestamp=T0)
        assert svc.total_events == 2

    def test_event_id_sequential(self, svc):
        d1 = svc.process(_known("EMP001"), timestamp=T0)
        d2 = svc.process(_known("EMP002"), timestamp=T0)
        assert d2.event.event_id == d1.event.event_id + 1


# ---------------------------------------------------------------------------
# Test: Cooldown chống ghi lặp
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_second_recognition_within_cooldown_skipped(self, svc):
        svc.process(_known(), timestamp=T0)
        d2 = svc.process(_known(), timestamp=T0 + 10.0)   # 10s < 30s cooldown
        assert d2.created is False
        assert d2.skip_reason == SkipReason.COOLDOWN

    def test_cooldown_remaining_calculated(self, svc):
        svc.process(_known(), timestamp=T0)
        d2 = svc.process(_known(), timestamp=T0 + 10.0)
        assert abs(d2.cooldown_remaining - 20.0) < 0.1

    def test_after_cooldown_new_event_created(self, svc):
        """Vượt cooldown → không bị skip cooldown, nhưng có thể bị min_toggle nếu loại sự kiện tiếp là CHECK_OUT.
        Dùng timestamp vượt cả min_toggle (60s) để đảm bảo sự kiện được tạo."""
        svc.process(_known(), timestamp=T0)
        d2 = svc.process(_known(), timestamp=T0 + 61.0)   # 61s > min_toggle(60) > cooldown(30)
        assert d2.created is True

    def test_different_employees_independent_cooldown(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)
        d2 = svc.process(_known("EMP002"), timestamp=T0 + 1.0)
        assert d2.created is True  # EMP002 chưa có cooldown

    def test_different_cameras_independent_cooldown(self, svc):
        svc.process(_known("EMP001"), camera_id="cam_0", timestamp=T0)
        d2 = svc.process(_known("EMP001"), camera_id="cam_1", timestamp=T0 + 5.0)
        assert d2.created is True  # Camera khác → không bị cooldown

    def test_no_cooldown_on_very_first_event(self, svc):
        d = svc.process(_known(), timestamp=T0)
        assert d.created is True


# ---------------------------------------------------------------------------
# Test: Luân phiên CHECK_IN / CHECK_OUT
# ---------------------------------------------------------------------------

class TestToggle:
    def test_sequence_in_out_in_out(self, svc):
        gap = 65.0  # > min_toggle=60s và > cooldown=30s

        d1 = svc.process(_known(), timestamp=T0)
        assert d1.event.event_type == EventType.CHECK_IN

        d2 = svc.process(_known(), timestamp=T0 + gap)
        assert d2.event.event_type == EventType.CHECK_OUT

        d3 = svc.process(_known(), timestamp=T0 + gap * 2)
        assert d3.event.event_type == EventType.CHECK_IN

        d4 = svc.process(_known(), timestamp=T0 + gap * 3)
        assert d4.event.event_type == EventType.CHECK_OUT

    def test_check_out_before_min_toggle_skipped(self, svc):
        svc.process(_known(), timestamp=T0)                   # CHECK_IN
        d2 = svc.process(_known(), timestamp=T0 + 35.0)       # 35s > cooldown(30) nhưng < min_toggle(60)
        assert d2.created is False
        assert d2.skip_reason == SkipReason.MIN_TOGGLE

    def test_check_out_after_min_toggle_ok(self, svc):
        svc.process(_known(), timestamp=T0)                   # CHECK_IN
        d2 = svc.process(_known(), timestamp=T0 + 61.0)       # 61s > min_toggle(60)
        assert d2.created is True
        assert d2.event.event_type == EventType.CHECK_OUT

    def test_min_toggle_message_informative(self, svc):
        svc.process(_known(), timestamp=T0)
        d2 = svc.process(_known(), timestamp=T0 + 35.0)
        assert d2.message  # không rỗng
        assert "CHECK_IN" in d2.message

    def test_check_in_after_check_out_no_min_toggle(self, svc):
        """CHECK_OUT → CHECK_IN tiếp theo chỉ cần qua cooldown, không cần min_toggle."""
        svc.process(_known(), timestamp=T0)                   # CHECK_IN
        svc.process(_known(), timestamp=T0 + 61.0)            # CHECK_OUT
        d3 = svc.process(_known(), timestamp=T0 + 61.0 + 31.0)  # CHECK_IN tiếp
        assert d3.created is True
        assert d3.event.event_type == EventType.CHECK_IN


# ---------------------------------------------------------------------------
# Test: nhiều nhân viên / nhiều camera cùng lúc
# ---------------------------------------------------------------------------

class TestMultipleEntities:
    def test_two_employees_independent(self, svc):
        svc.process(_known("EMP001"), camera_id="cam_0", timestamp=T0)
        svc.process(_known("EMP002"), camera_id="cam_0", timestamp=T0 + 1.0)
        assert svc.total_events == 2

    def test_same_employee_two_cameras(self, svc):
        svc.process(_known("EMP001"), camera_id="cam_0", timestamp=T0)
        svc.process(_known("EMP001"), camera_id="cam_1", timestamp=T0 + 2.0)
        assert svc.total_events == 2

    def test_events_for_employee_filter(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)
        svc.process(_known("EMP002"), timestamp=T0)
        svc.process(_known("EMP001"), timestamp=T0 + 70.0)   # CHECK_OUT EMP001

        events_emp1 = svc.events_for(employee_id="EMP001")
        assert len(events_emp1) == 2

    def test_events_for_event_type_filter(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)             # CHECK_IN
        svc.process(_known("EMP001"), timestamp=T0 + 70.0)      # CHECK_OUT
        svc.process(_known("EMP002"), timestamp=T0)             # CHECK_IN

        checkins = svc.events_for(event_type=EventType.CHECK_IN)
        assert len(checkins) == 2

    def test_events_for_camera_filter(self, svc):
        svc.process(_known("EMP001"), camera_id="cam_0", timestamp=T0)
        svc.process(_known("EMP002"), camera_id="cam_1", timestamp=T0)
        events_cam0 = svc.events_for(camera_id="cam_0")
        assert len(events_cam0) == 1

    def test_last_event_for_employee(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)             # CHECK_IN
        svc.process(_known("EMP001"), timestamp=T0 + 70.0)      # CHECK_OUT
        last = svc.last_event_for("EMP001")
        assert last is not None
        assert last.event_type == EventType.CHECK_OUT

    def test_last_event_for_nonexistent(self, svc):
        assert svc.last_event_for("GHOST") is None


# ---------------------------------------------------------------------------
# Test: reset_state
# ---------------------------------------------------------------------------

class TestResetState:
    def test_reset_all_clears_cooldown(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)
        svc.reset_state()
        # Sau reset, CHECK_IN lại ngay được
        d = svc.process(_known("EMP001"), timestamp=T0 + 1.0)
        assert d.created is True
        assert d.event.event_type == EventType.CHECK_IN

    def test_reset_specific_employee(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)
        svc.process(_known("EMP002"), timestamp=T0)
        svc.reset_state(employee_id="EMP001")

        # EMP001 bị reset → CHECK_IN lại
        d1 = svc.process(_known("EMP001"), timestamp=T0 + 1.0)
        assert d1.created is True
        assert d1.event.event_type == EventType.CHECK_IN

        # EMP002 vẫn còn cooldown
        d2 = svc.process(_known("EMP002"), timestamp=T0 + 1.0)
        assert d2.created is False
        assert d2.skip_reason == SkipReason.COOLDOWN

    def test_reset_does_not_clear_history(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)
        total_before = svc.total_events
        svc.reset_state()
        assert svc.total_events == total_before


# ---------------------------------------------------------------------------
# Test: clear_history
# ---------------------------------------------------------------------------

class TestClearHistory:
    def test_clear_history_empties_events(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)
        svc.clear_history()
        assert svc.total_events == 0

    def test_clear_history_resets_event_id(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)
        svc.clear_history()
        d = svc.process(_known("EMP001"), timestamp=T0 + 100.0)
        assert d.event.event_id == 1

    def test_clear_history_does_not_clear_state(self, svc):
        """Cooldown vẫn còn sau clear_history."""
        svc.process(_known("EMP001"), timestamp=T0)
        svc.clear_history()
        d = svc.process(_known("EMP001"), timestamp=T0 + 1.0)
        assert d.created is False   # vẫn trong cooldown


# ---------------------------------------------------------------------------
# Test: cooldown_remaining helper
# ---------------------------------------------------------------------------

class TestCooldownRemaining:
    def test_no_state_returns_zero(self, svc):
        assert svc.cooldown_remaining("EMP001") == 0.0

    def test_within_cooldown_returns_positive(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)
        # Mock time.time để trả về T0 + 10
        with patch("app.attendance.attendance_service.time.time", return_value=T0 + 10.0):
            remaining = svc.cooldown_remaining("EMP001")
        assert abs(remaining - 20.0) < 0.1

    def test_after_cooldown_returns_zero(self, svc):
        svc.process(_known("EMP001"), timestamp=T0)
        with patch("app.attendance.attendance_service.time.time", return_value=T0 + 100.0):
            remaining = svc.cooldown_remaining("EMP001")
        assert remaining == 0.0


# ---------------------------------------------------------------------------
# Test: AttendanceEvent helpers
# ---------------------------------------------------------------------------

class TestAttendanceEvent:
    def _make_event(self) -> AttendanceEvent:
        return AttendanceEvent(
            event_id=1,
            employee_id="EMP001",
            event_type=EventType.CHECK_IN,
            timestamp_utc=T0,
            camera_id="cam_0",
            similarity=0.65,
        )

    def test_to_local_dt_timezone(self):
        event = self._make_event()
        tz = timezone(timedelta(hours=7))
        dt = event.to_local_dt(tz)
        assert dt.utcoffset() == timedelta(hours=7)

    def test_str_contains_employee_id(self):
        event = self._make_event()
        assert "EMP001" in str(event)

    def test_str_contains_event_type(self):
        event = self._make_event()
        assert "CHECK_IN" in str(event)

    def test_to_dict_has_required_keys(self):
        event = self._make_event()
        d = event.to_dict()
        for key in ["event_id", "employee_id", "event_type", "timestamp_utc",
                    "timestamp_local", "camera_id", "similarity"]:
            assert key in d

    def test_to_dict_event_type_is_string(self):
        event = self._make_event()
        d = event.to_dict()
        assert isinstance(d["event_type"], str)


# ---------------------------------------------------------------------------
# Test: Export JSON / CSV
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_json(self, svc, tmp_path):
        svc.process(_known("EMP001"), timestamp=T0)
        svc.process(_known("EMP002"), timestamp=T0)
        filepath = str(tmp_path / "attendance.json")
        svc.export_json(filepath)

        import json
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["employee_id"] == "EMP001"

    def test_export_csv(self, svc, tmp_path):
        svc.process(_known("EMP001"), timestamp=T0)
        filepath = str(tmp_path / "attendance.csv")
        svc.export_csv(filepath)

        import csv
        with open(filepath, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["employee_id"] == "EMP001"
        assert rows[0]["event_type"] == "check_in"

    def test_export_json_empty(self, svc, tmp_path):
        filepath = str(tmp_path / "empty.json")
        svc.export_json(filepath)
        import json
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        assert data == []


# ---------------------------------------------------------------------------
# Test: Tình huống biên
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_exactly_at_cooldown_boundary_accepted(self, svc):
        """elapsed == cooldown → qua cooldown, nhưng CHECK_OUT còn cần min_toggle.
        Test này xác nhận cooldown NOT triggered; min_toggle có thể vẫn chặn.
        Dùng timestamp vượt min_toggle để test thuần túy boundary của cooldown."""
        svc.process(_known(), timestamp=T0)
        d = svc.process(_known(), timestamp=T0 + 60.0)
        # 60s >= cooldown(30) VÀ 60s >= min_toggle(60) → được tạo CHECK_OUT
        assert d.created is True
        assert d.skip_reason != SkipReason.COOLDOWN

    def test_exactly_at_min_toggle_boundary_accepted(self, svc):
        svc.process(_known(), timestamp=T0)                  # CHECK_IN
        d = svc.process(_known(), timestamp=T0 + 60.0)       # elapsed == min_toggle=60 → OK
        assert d.created is True
        assert d.event.event_type == EventType.CHECK_OUT

    def test_many_events_accumulate(self, svc):
        gap = 70.0
        for i in range(6):
            svc.process(_known("EMP001"), timestamp=T0 + i * gap)
        assert svc.total_events == 6

    def test_decision_employee_id_property(self, svc):
        d = svc.process(_known("EMP001"), timestamp=T0)
        assert d.employee_id == "EMP001"

    def test_decision_no_event_employee_id_none(self, svc):
        d = svc.process(_unknown(), timestamp=T0)
        assert d.employee_id is None

    def test_process_uses_current_time_when_no_timestamp(self, svc):
        """Khi không truyền timestamp, dùng time.time() thực."""
        before = time.time()
        d = svc.process(_known("EMP001"))
        after = time.time()
        assert d.created is True
        assert before <= d.event.timestamp_utc <= after
