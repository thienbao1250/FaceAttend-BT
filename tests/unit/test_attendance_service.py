"""
tests/unit/test_attendance_service.py
Unit & Integration tests cho AttendanceService — Giai đoạn 11.

Kiểm tra đầy đủ:
  - Quy trình nhận diện, cooldown, lọc nhân viên active
  - Trạng thái IN_PROGRESS / COMPLETED / INCOMPLETE
  - 11 chỉ số thời gian làm việc (TotalPresence, Break, ActualWork, Missing, Extra, Overtime)
  - Tất cả 15 trường hợp ngoại lệ trong Mục 7
  - Thread-safety khi nhiều yêu cầu nhận diện gửi đồng thời
  - Xuất dữ liệu phân cấp theo Năm / Tháng / Ngày
"""

from __future__ import annotations

import csv
import json
import threading
import time
from datetime import date, datetime, time as dt_time, timedelta, timezone
from unittest.mock import patch

import pytest

from app.attendance.attendance_service import (
    AttendanceDecision,
    AttendanceEvent,
    AttendanceService,
    AttendanceStatus,
    EventType,
    PunctualityStatus,
    ShiftConfig,
    SkipReason,
)
from app.vision.face_matcher import MatchResult, MatchStatus

VN_TZ = timezone(timedelta(hours=7))


def _epoch(y: int, m: int, d: int, hh: int, mm: int, ss: int = 0) -> float:
    return datetime(y, m, d, hh, mm, ss, tzinfo=VN_TZ).timestamp()


def _known(emp_id: str = "EMP001", sim: float = 0.75) -> MatchResult:
    return MatchResult(
        status=MatchStatus.KNOWN,
        employee_id=emp_id,
        similarity=sim,
        num_candidates=1,
    )


def _unknown(sim: float = 0.05) -> MatchResult:
    return MatchResult(status=MatchStatus.UNKNOWN, employee_id=None, similarity=sim, num_candidates=1)


def _uncertain(sim: float = 0.28) -> MatchResult:
    return MatchResult(status=MatchStatus.UNCERTAIN, employee_id=None, similarity=sim, num_candidates=1)


T_CHECKIN_ONTIME = _epoch(2026, 11, 26, 8, 0, 0)


@pytest.fixture
def svc():
    """Ca hành chính chuẩn 08:00–17:00, nghỉ trưa 12:00–13:00 (60p), grace 5p."""
    return AttendanceService(
        cooldown_seconds=30.0,
        min_toggle_seconds=60.0,
        shift=ShiftConfig(
            shift_code="HC_08_17",
            start_time=dt_time(8, 0),
            end_time=dt_time(17, 0),
            break_start=dt_time(12, 0),
            break_end=dt_time(13, 0),
            break_minutes=60,
            standard_work_minutes=480,
            grace_period_minutes=5,
            early_check_in_minutes=120,
            late_check_out_minutes=240,
            checkout_window_before_end_minutes=180,
            missing_checkin_cutoff_minutes=180,
            is_overnight=False,
        ),
    )


# ---------------------------------------------------------------------------
# 15 trường hợp ngoại lệ (Mục 7)
# ---------------------------------------------------------------------------

class TestFifteenExceptionCases:

    def test_case_01_check_in_on_time(self, svc):
        """1. Nhân viên nhận diện thành công và Check-in đúng giờ (trong grace 5p)."""
        d = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 5, 0))
        assert d.created is True
        assert d.event.event_type == EventType.CHECK_IN
        assert d.event.attendance_status == AttendanceStatus.IN_PROGRESS
        assert d.event.punctuality_status == PunctualityStatus.ON_TIME
        assert d.event.late_minutes == 0

    def test_case_02_check_in_late(self, svc):
        """2. Nhân viên Check-in trễ (08:20 -> trễ 20 phút)."""
        d = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 20, 0))
        assert d.created is True
        assert d.event.event_type == EventType.CHECK_IN
        assert d.event.attendance_status == AttendanceStatus.IN_PROGRESS
        assert d.event.punctuality_status == PunctualityStatus.LATE
        assert d.event.late_minutes == 20

    def test_case_03_check_out_on_time(self, svc):
        """3. Nhân viên Check-out đúng giờ (08:00 -> 17:00) => COMPLETED, 8h làm việc."""
        svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 0, 0))
        d = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 17, 0, 0))
        assert d.created is True
        assert d.event.event_type == EventType.CHECK_OUT
        assert d.event.attendance_status == AttendanceStatus.COMPLETED
        assert d.event.early_leave_minutes == 0
        rec = d.record
        assert rec.total_presence_minutes == 540
        assert rec.break_minutes == 60
        assert rec.actual_work_minutes == 480
        assert rec.missing_minutes == 0
        assert rec.extra_minutes == 0

    def test_case_04_check_out_early_leave(self, svc):
        """4. Nhân viên về sớm (08:00 -> 16:00 -> về sớm 60 phút, thiếu 60 phút)."""
        svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 0, 0))
        d = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 16, 0, 0))
        assert d.created is True
        assert d.event.event_type == EventType.CHECK_OUT
        assert d.event.early_leave_minutes == 60
        assert d.record.punctuality_status == PunctualityStatus.EARLY_LEAVE
        assert d.record.actual_work_minutes == 420
        assert d.record.missing_minutes == 60

    def test_case_05_checkin_without_checkout_reconciled_to_incomplete(self, svc):
        """5. Đã Check-in nhưng không Check-out -> IN_PROGRESS trong ca; chỉ INCOMPLETE sau đối soát cuối ca."""
        svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 0, 0))
        rec_mid = svc.get_record("EMP001", date(2026, 11, 26))
        assert rec_mid.attendance_status == AttendanceStatus.IN_PROGRESS

        # Chưa hết khung ca (18:00 < 17:00 + 240p=21:00) -> không chuyển INCOMPLETE
        assert svc.reconcile_shift(date(2026, 11, 26), _epoch(2026, 11, 26, 18, 0, 0)) == []
        assert rec_mid.attendance_status == AttendanceStatus.IN_PROGRESS

        # Sau khi ca + late_check_out kết thúc (21:05) -> đối soát thành INCOMPLETE
        reconciled = svc.reconcile_shift(date(2026, 11, 26), _epoch(2026, 11, 26, 21, 5, 0))
        assert len(reconciled) == 1
        assert rec_mid.attendance_status == AttendanceStatus.INCOMPLETE
        assert rec_mid.needs_review is True
        assert rec_mid.review_reason == "MISSING_CHECK_OUT"

    def test_case_06_checkout_without_checkin_incomplete_needs_review(self, svc):
        """6. Không Check-in nhưng có Check-out (quét lần đầu lúc 17:00) -> INCOMPLETE, chờ HR đối soát."""
        d = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 17, 0, 0))
        assert d.created is True
        assert d.event.event_type == EventType.CHECK_OUT
        assert d.record.attendance_status == AttendanceStatus.INCOMPLETE
        assert d.record.check_in_time is None
        assert d.record.check_out_time is not None
        assert d.record.needs_review is True
        assert d.record.review_reason == "MISSING_CHECK_IN"

    def test_case_07_continuous_scans_blocked_by_cooldown(self, svc):
        """7. Nhân viên nhận diện nhiều lần liên tục (trong 30s) không tạo bản ghi trùng."""
        d1 = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 0, 0))
        d2 = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 0, 5))
        d3 = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 0, 20))
        assert d1.created is True
        assert d2.created is False and d2.skip_reason == SkipReason.COOLDOWN
        assert d3.created is False and d3.skip_reason == SkipReason.COOLDOWN
        assert svc.total_events == 1

    def test_case_08_lunch_break_exit_and_return_does_not_checkout_or_recheckin(self, svc):
        """8. Ra ngoài ăn trưa (12:05) rồi quay lại (12:50) KHÔNG được tạo Check-out và Check-in mới."""
        d_in = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 0, 0))
        d_lunch_out = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 12, 5, 0))
        d_lunch_back = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 12, 50, 0))

        assert d_in.created is True
        assert d_lunch_out.created is False
        assert d_lunch_out.skip_reason == SkipReason.ALREADY_CHECKED_IN
        assert d_lunch_back.created is False
        assert d_lunch_back.skip_reason == SkipReason.ALREADY_CHECKED_IN

        rec = svc.get_record("EMP001", date(2026, 11, 26))
        assert rec.attendance_status == AttendanceStatus.IN_PROGRESS
        assert rec.check_out_time is None
        assert svc.total_events == 1

    def test_case_09_outside_shift_hours_rejected(self, svc):
        """9. Chấm công ngoài khung giờ làm việc (04:00 sáng hoặc 22:30 tối) bị từ chối."""
        d_early = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 4, 0, 0))
        d_late = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 22, 30, 0))
        assert d_early.created is False and d_early.skip_reason == SkipReason.OUTSIDE_SHIFT_HOURS
        assert d_late.created is False and d_late.skip_reason == SkipReason.OUTSIDE_SHIFT_HOURS

    def test_case_10_completed_employee_standing_in_front_of_camera(self, svc):
        """10. Đã hoàn thành Check-in & Check-out, tiếp tục đứng trước camera không tạo thêm bản ghi."""
        svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 0, 0))
        d_out = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 17, 0, 0))
        d_extra = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 17, 5, 0))
        assert d_out.created is True
        assert d_extra.created is False
        assert d_extra.skip_reason == SkipReason.ALREADY_COMPLETED
        assert svc.total_events == 2

    def test_case_11_unknown_or_uncertain_face_not_recorded(self, svc):
        """11. Camera không nhận diện được (UNKNOWN / UNCERTAIN) không tạo bản ghi."""
        d1 = svc.process(_unknown(), timestamp=T_CHECKIN_ONTIME)
        d2 = svc.process(_uncertain(), timestamp=T_CHECKIN_ONTIME)
        assert d1.created is False and d1.skip_reason == SkipReason.UNKNOWN
        assert d2.created is False and d2.skip_reason == SkipReason.UNCERTAIN
        assert svc.total_events == 0

    def test_case_12_inactive_or_nonexistent_employee_rejected(self, svc):
        """12. Nhân viên không tồn tại trong danh sách active hoặc đã bị vô hiệu hóa bị chặn."""
        svc.set_active_employees({"EMP001"})
        d_inactive_list = svc.process(_known("EMP999"), timestamp=T_CHECKIN_ONTIME)
        d_inactive_flag = svc.process(_known("EMP001"), timestamp=T_CHECKIN_ONTIME, is_employee_active=False)
        assert d_inactive_list.created is False and d_inactive_list.skip_reason == SkipReason.EMPLOYEE_INACTIVE
        assert d_inactive_flag.created is False and d_inactive_flag.skip_reason == SkipReason.EMPLOYEE_INACTIVE
        assert svc.total_events == 0

    def test_case_13_overnight_shift_cross_midnight(self):
        """13. Ca đêm qua ngày (22:00 hôm nay -> 06:00 hôm sau), work_date tính theo ngày bắt đầu ca."""
        night_svc = AttendanceService(
            shift=ShiftConfig(
                shift_code="CA_DEM_22_06",
                start_time=dt_time(22, 0),
                end_time=dt_time(6, 0),
                break_start=None,
                break_end=None,
                break_minutes=0,
                standard_work_minutes=480,
                grace_period_minutes=5,
                is_overnight=True,
            )
        )
        d_in = night_svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 22, 0, 0))
        d_out = night_svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 27, 6, 0, 0))
        assert d_in.created is True and d_in.event.work_date == date(2026, 11, 26)
        assert d_out.created is True and d_out.event.work_date == date(2026, 11, 26)
        rec = night_svc.get_record("EMP001", date(2026, 11, 26))
        assert rec.attendance_status == AttendanceStatus.COMPLETED
        assert rec.actual_work_minutes == 480

    def test_case_14_actual_work_less_than_standard(self, svc):
        """14. Làm thực tế thấp hơn tiêu chuẩn (09:00 -> 16:00 => 6h = 360p, thiếu 120p)."""
        svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 9, 0, 0))
        d = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 16, 0, 0))
        rec = d.record
        assert rec.total_presence_minutes == 420
        assert rec.break_minutes == 60
        assert rec.actual_work_minutes == 360
        assert rec.missing_minutes == 120
        assert rec.extra_minutes == 0

    def test_case_15_extra_minutes_separated_from_approved_overtime(self, svc):
        """15. Làm vượt tiêu chuẩn (08:00 -> 18:00 => 9h = 540p): ExtraMinutes=60, OvertimeMinutes=0 nếu chưa duyệt."""
        svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 0, 0))
        d = svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 18, 0, 0))
        rec = d.record
        assert rec.actual_work_minutes == 540
        assert rec.extra_minutes == 60
        assert rec.overtime_minutes == 0
        assert rec.overtime_approved is False

        # Chỉ khi được duyệt tăng ca mới ghi nhận OvertimeMinutes
        svc.approve_overtime("EMP001", date(2026, 11, 26), 60)
        assert rec.overtime_minutes == 60
        assert rec.overtime_approved is True


# ---------------------------------------------------------------------------
# Concurrency, Helpers & Export
# ---------------------------------------------------------------------------

class TestConcurrencyAndHelpers:

    def test_concurrent_same_employee_creates_only_one_checkin(self, svc):
        """Gửi 10 luồng nhận diện đồng thời cho cùng 1 NV -> chỉ tạo đúng 1 CHECK_IN."""
        results: list[AttendanceDecision] = []
        lock = threading.Lock()

        def worker():
            res = svc.process(_known("EMP_CONC"), timestamp=T_CHECKIN_ONTIME)
            with lock:
                results.append(res)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        created_count = sum(1 for r in results if r.created)
        assert created_count == 1
        assert svc.total_events == 1

    def test_different_employees_and_cameras(self, svc):
        svc.process(_known("EMP001"), camera_id="cam_0", timestamp=T_CHECKIN_ONTIME)
        svc.process(_known("EMP002"), camera_id="cam_0", timestamp=T_CHECKIN_ONTIME + 1.0)
        assert svc.total_events == 2
        assert len(svc.events_for(employee_id="EMP001")) == 1
        assert svc.last_event_for("GHOST") is None

    def test_cooldown_remaining(self, svc):
        svc.process(_known("EMP001"), timestamp=T_CHECKIN_ONTIME)
        with patch("app.attendance.attendance_service.time_mod.time", return_value=T_CHECKIN_ONTIME + 10.0):
            assert abs(svc.cooldown_remaining("EMP001") - 20.0) < 0.1
        with patch("app.attendance.attendance_service.time_mod.time", return_value=T_CHECKIN_ONTIME + 40.0):
            assert svc.cooldown_remaining("EMP001") == 0.0

    def test_export_by_date_creates_year_month_day_hierarchy(self, svc, tmp_path):
        svc.process(_known("EMP001"), timestamp=_epoch(2026, 11, 26, 8, 0, 0))
        svc.process(_known("EMP002"), timestamp=_epoch(2026, 11, 27, 8, 10, 0))

        dirs = svc.export_by_date(tmp_path)
        assert len(dirs) == 2
        day1 = tmp_path / "2026" / "11" / "26"
        day2 = tmp_path / "2026" / "11" / "27"
        assert (day1 / "attendance.json").exists()
        assert (day1 / "attendance.csv").exists()
        assert (day2 / "attendance.json").exists()

        data1 = json.loads((day1 / "attendance.json").read_text(encoding="utf-8"))
        assert data1[0]["employee_id"] == "EMP001"
        assert data1[0]["attendance_status"] == "IN_PROGRESS"

        with open(day1 / "attendance.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["verify_method"] == "Face Recognition"
        assert rows[0]["attendance_status"] == "IN_PROGRESS"

    def test_to_display_dict_has_business_fields(self):
        ev = AttendanceEvent(
            event_id=100001,
            employee_id="FA-IT-2026-0001",
            event_type=EventType.CHECK_IN,
            timestamp_utc=_epoch(2026, 11, 26, 8, 5, 30),
            camera_id="CAM001",
            similarity=0.92,
            attendance_status=AttendanceStatus.IN_PROGRESS,
            punctuality_status=PunctualityStatus.ON_TIME,
            late_minutes=0,
        )
        d = ev.to_display_dict()
        assert d["Employee ID"] == "FA-IT-2026-0001"
        assert d["Thời gian"] == "2026-11-26 08:05:30"
        assert d["Device ID"] == "CAM001"
        assert d["Phương thức"] == "Face Recognition"
        assert d["Event Type"] == "Check-in"
        assert d["Verify Status"] == "Success"
        assert d["Result"] == "Accepted"
        assert d["Record ID"] == 100001
        assert d["Trạng thái ca"] == "IN_PROGRESS"
