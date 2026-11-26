"""
app/attendance/attendance_service.py
Logic chấm công theo ca làm việc — Giai đoạn 11.

Chuyển kết quả nhận diện khuôn mặt (MatchResult) thành sự kiện chấm công
(AttendanceEvent) và bản ghi chấm công theo ca (AttendanceRecordState).

Quy tắc nghiệp vụ cốt lõi:
  1. Kiểm tra nhân viên hợp lệ (đã đăng ký & đang hoạt động) và ngưỡng confidence.
  2. Cooldown chống ghi lặp khi đứng trước camera nhiều frame liên tiếp.
  3. Xác định CHECK_IN / CHECK_OUT dựa trên ca làm việc (ShiftConfig), thời gian
     hiện tại và trạng thái chấm công hiện tại (IN_PROGRESS / COMPLETED / INCOMPLETE):
       - Không tự động Check-out khi ra ngoài ăn trưa rồi quay lại.
       - Đã Check-in (IN_PROGRESS) chỉ Check-out khi vào khung giờ Check-out hợp lệ
         (hoặc có chỉ định explicit_event_type=EventType.CHECK_OUT).
       - Đã COMPLETED thì không tạo thêm bản ghi cho cùng ca.
       - Hỗ trợ ca qua ngày / ca đêm (is_overnight=True).
       - Hỗ trợ đối soát cuối ca (reconcile_shift) chuyển IN_PROGRESS -> INCOMPLETE.
  4. Tính toán đầy đủ 11 chỉ số thời gian làm việc:
       CheckInTime, CheckOutTime, TotalPresenceMinutes, BreakMinutes,
       ActualWorkMinutes, StandardWorkMinutes, LateMinutes, EarlyLeaveMinutes,
       MissingMinutes, ExtraMinutes, OvertimeMinutes (chỉ ghi nhận khi được duyệt).
  5. Thread-safe (RLock) khi nhiều yêu cầu nhận diện gửi đồng thời.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time as time_mod
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from app.vision.face_matcher import MatchResult, MatchStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hằng số / ngưỡng mặc định
# ---------------------------------------------------------------------------

DEFAULT_COOLDOWN_SECONDS: float = 30.0
DEFAULT_MIN_TOGGLE_SECONDS: float = 60.0
DEFAULT_TIMEZONE = timezone(timedelta(hours=7))

DEFAULT_SHIFT_START_HOUR: int = 8
DEFAULT_SHIFT_START_MINUTE: int = 0
DEFAULT_SHIFT_END_HOUR: int = 17
DEFAULT_SHIFT_END_MINUTE: int = 0
DEFAULT_LATE_THRESHOLD_MINUTES: int = 5
DEFAULT_EARLY_LEAVE_THRESHOLD_MINUTES: int = 5


# ---------------------------------------------------------------------------
# Enums nghiệp vụ
# ---------------------------------------------------------------------------

class EventType(Enum):
    CHECK_IN  = "check_in"
    CHECK_OUT = "check_out"


class AttendanceStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"  # Đã Check-in nhưng chưa Check-out
    COMPLETED   = "COMPLETED"    # Đã có cả Check-in và Check-out
    INCOMPLETE  = "INCOMPLETE"   # Thiếu Check-in hoặc Check-out sau khi ca kết thúc & đối soát


class PunctualityStatus(Enum):
    ON_TIME        = "ON_TIME"
    LATE           = "LATE"
    EARLY_LEAVE    = "EARLY_LEAVE"
    LATE_AND_EARLY = "LATE_AND_EARLY"
    UNKNOWN        = "UNKNOWN"

    @property
    def display_vi(self) -> str:
        return {
            "ON_TIME":        "✅ Đúng giờ",
            "LATE":           "⚠️  Trễ giờ",
            "EARLY_LEAVE":    "⏰ Về sớm",
            "LATE_AND_EARLY": "⚠️  Trễ & Về sớm",
            "UNKNOWN":        "—  Không rõ",
        }[self.value]


class SkipReason(Enum):
    COOLDOWN            = "cooldown"             # Chưa hết thời gian chờ chống ghi lặp
    UNCERTAIN           = "uncertain"            # Nhận diện không chắc chắn
    UNKNOWN             = "unknown"              # Người chưa đăng ký
    EMPLOYEE_INACTIVE   = "employee_inactive"    # Nhân viên bị vô hiệu hóa / đã nghỉ
    OUTSIDE_SHIFT_HOURS = "outside_shift_hours"  # Ngoài khung giờ hợp lệ của ca
    ALREADY_CHECKED_IN  = "already_checked_in"   # Đã Check-in, chưa đến khung Check-out
    ALREADY_COMPLETED   = "already_completed"    # Đã hoàn thành chấm công ca này
    MIN_TOGGLE          = "min_toggle"           # Chưa đủ thời gian tối thiểu


# ---------------------------------------------------------------------------
# Cấu hình ca làm việc
# ---------------------------------------------------------------------------

@dataclass
class ShiftConfig:
    """Cấu hình một ca làm việc chuẩn."""
    shift_code:                         str            = "HC_08_17"
    name:                               str            = "Ca hành chính (08:00–17:00)"
    start_time:                         time           = field(default_factory=lambda: time(8, 0))
    end_time:                           time           = field(default_factory=lambda: time(17, 0))
    break_start:                        Optional[time] = field(default_factory=lambda: time(12, 0))
    break_end:                          Optional[time] = field(default_factory=lambda: time(13, 0))
    break_minutes:                      int            = 60
    standard_work_minutes:              int            = 480
    grace_period_minutes:               int            = 5
    early_leave_grace_minutes:          int            = 0
    early_check_in_minutes:             int            = 120
    late_check_out_minutes:             int            = 240
    checkout_window_before_end_minutes: int            = 180
    missing_checkin_cutoff_minutes:     int            = 180
    is_overnight:                       bool           = False

    def resolve_shift_window(
        self, local_dt: datetime
    ) -> tuple[date, datetime, datetime, Optional[datetime], Optional[datetime]]:
        """
        Xác định (work_date, shift_start_dt, shift_end_dt, break_start_dt, break_end_dt)
        tại thời điểm local_dt (có xử lý ca đêm / ca qua ngày).
        """
        tz = local_dt.tzinfo
        today = local_dt.date()

        if self.is_overnight:
            # Nếu đang ở nửa sau của ca đêm (sau 00:00 đến end_time + late_check_out),
            # ngày làm việc (work_date) là ngày hôm qua (lúc ca bắt đầu).
            end_today = datetime.combine(today, self.end_time, tzinfo=tz)
            cutoff_today = end_today + timedelta(minutes=self.late_check_out_minutes)
            if local_dt <= cutoff_today:
                work_date = today - timedelta(days=1)
            else:
                work_date = today
            shift_start_dt = datetime.combine(work_date, self.start_time, tzinfo=tz)
            shift_end_dt = datetime.combine(work_date + timedelta(days=1), self.end_time, tzinfo=tz)
        else:
            work_date = today
            shift_start_dt = datetime.combine(work_date, self.start_time, tzinfo=tz)
            shift_end_dt = datetime.combine(work_date, self.end_time, tzinfo=tz)

        break_start_dt: Optional[datetime] = None
        break_end_dt: Optional[datetime] = None
        if self.break_start is not None and self.break_end is not None:
            bs_date = work_date
            be_date = work_date
            if self.is_overnight:
                if self.break_start < self.start_time:
                    bs_date = work_date + timedelta(days=1)
                if self.break_end < self.start_time:
                    be_date = work_date + timedelta(days=1)
            break_start_dt = datetime.combine(bs_date, self.break_start, tzinfo=tz)
            break_end_dt = datetime.combine(be_date, self.break_end, tzinfo=tz)

        return work_date, shift_start_dt, shift_end_dt, break_start_dt, break_end_dt


# ---------------------------------------------------------------------------
# Bản ghi trạng thái ca làm việc theo ngày của nhân viên
# ---------------------------------------------------------------------------

@dataclass
class AttendanceRecordState:
    """Lưu trạng thái & 11 chỉ số thời gian làm việc của một nhân viên trong một ca."""
    employee_id:            str
    work_date:              date
    shift_code:             str
    shift_start_utc:        datetime
    shift_end_utc:          datetime
    break_start_utc:        Optional[datetime] = None
    break_end_utc:          Optional[datetime] = None
    attendance_status:      AttendanceStatus   = AttendanceStatus.IN_PROGRESS
    punctuality_status:     PunctualityStatus  = PunctualityStatus.UNKNOWN
    check_in_time:          Optional[datetime] = None
    check_out_time:         Optional[datetime] = None
    total_presence_minutes: Optional[int]      = None
    break_minutes:          int                = 0
    actual_work_minutes:    Optional[int]      = None
    standard_work_minutes:  int                = 480
    late_minutes:           int                = 0
    early_leave_minutes:    int                = 0
    missing_minutes:        int                = 0
    extra_minutes:          int                = 0
    overtime_minutes:       int                = 0
    overtime_approved:      bool               = False
    needs_review:           bool               = False
    review_reason:          Optional[str]      = None


# ---------------------------------------------------------------------------
# Sự kiện chấm công
# ---------------------------------------------------------------------------

@dataclass
class AttendanceEvent:
    """Một sự kiện chấm công hợp lệ (CHECK_IN hoặc CHECK_OUT)."""
    event_id:           int
    employee_id:        str
    event_type:         EventType
    timestamp_utc:      float
    camera_id:          str                = "cam_0"
    similarity:         float              = 0.0
    work_date:          Optional[date]     = None
    shift_code:         str                = "HC_08_17"
    attendance_status:  AttendanceStatus   = AttendanceStatus.IN_PROGRESS
    punctuality_status: PunctualityStatus  = PunctualityStatus.UNKNOWN
    late_minutes:       Optional[int]      = None
    early_leave_minutes: Optional[int]     = None
    actual_work_minutes: Optional[int]     = None
    verify_method:      str                = "Face Recognition"
    verify_status:      str                = "Success"
    result:             str                = "Accepted"

    def to_local_dt(self, tz: timezone = DEFAULT_TIMEZONE) -> datetime:
        return datetime.fromtimestamp(self.timestamp_utc, tz=tz)

    def __str__(self) -> str:
        dt = self.to_local_dt()
        punct = (
            self.punctuality_status.display_vi
            if self.punctuality_status != PunctualityStatus.UNKNOWN
            else ""
        )
        return (
            f"[{self.event_type.value.upper()}] "
            f"{self.employee_id} | "
            f"{dt.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"sim={self.similarity:.3f} | "
            f"cam={self.camera_id} | "
            f"{self.attendance_status.value}"
            + (f" | {punct}" if punct else "")
        )

    def to_dict(self) -> dict:
        return {
            "event_id":            self.event_id,
            "employee_id":         self.employee_id,
            "event_type":          self.event_type.value,
            "timestamp_utc":       self.timestamp_utc,
            "timestamp_local":     self.to_local_dt().isoformat(),
            "work_date":           str(self.work_date) if self.work_date else None,
            "shift_code":          self.shift_code,
            "camera_id":           self.camera_id,
            "similarity":          round(self.similarity, 4),
            "attendance_status":   self.attendance_status.value,
            "punctuality_status":  self.punctuality_status.value,
            "late_minutes":        self.late_minutes,
            "early_leave_minutes": self.early_leave_minutes,
            "actual_work_minutes": self.actual_work_minutes,
            "verify_method":       self.verify_method,
            "verify_status":       self.verify_status,
            "result":              self.result,
        }

    def to_display_dict(self, employee_code: str = "") -> dict:
        dt = self.to_local_dt()
        return {
            "Employee ID":   employee_code or self.employee_id,
            "Thời gian":     dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Device ID":     self.camera_id,
            "Phương thức":   self.verify_method,
            "Event Type":    "Check-in" if self.event_type == EventType.CHECK_IN else "Check-out",
            "Verify Status": self.verify_status,
            "Result":        self.result,
            "Record ID":     self.event_id,
            "Trạng thái":    self.punctuality_status.display_vi,
            "Trạng thái ca": self.attendance_status.value,
            "Đi trễ (phút)": self.late_minutes if self.late_minutes is not None else "—",
        }


# ---------------------------------------------------------------------------
# Kết quả xử lý một lần nhận diện
# ---------------------------------------------------------------------------

@dataclass
class AttendanceDecision:
    created:            bool
    event:              Optional[AttendanceEvent]       = None
    record:             Optional[AttendanceRecordState] = None
    skip_reason:        Optional[SkipReason]            = None
    message:            str                             = ""
    cooldown_remaining: float                           = 0.0

    @property
    def employee_id(self) -> Optional[str]:
        return self.event.employee_id if self.event else None


# ---------------------------------------------------------------------------
# Dịch vụ chấm công
# ---------------------------------------------------------------------------

class AttendanceService:
    """
    Áp dụng quy tắc nghiệp vụ chấm công theo ca làm việc.

    Hỗ trợ khởi tạo tương thích với các tham số cũ (shift_start_hour,
    late_threshold_minutes, ...) hoặc truyền trực tiếp `shift: ShiftConfig`.
    """

    def __init__(
        self,
        cooldown_seconds:              float                 = DEFAULT_COOLDOWN_SECONDS,
        min_toggle_seconds:            float                 = DEFAULT_MIN_TOGGLE_SECONDS,
        display_tz:                    timezone              = DEFAULT_TIMEZONE,
        shift_start_hour:              int                   = DEFAULT_SHIFT_START_HOUR,
        shift_start_minute:            int                   = DEFAULT_SHIFT_START_MINUTE,
        shift_end_hour:                int                   = DEFAULT_SHIFT_END_HOUR,
        shift_end_minute:              int                   = DEFAULT_SHIFT_END_MINUTE,
        late_threshold_minutes:        int                   = DEFAULT_LATE_THRESHOLD_MINUTES,
        early_leave_threshold_minutes: int                   = DEFAULT_EARLY_LEAVE_THRESHOLD_MINUTES,
        shift:                         Optional[ShiftConfig] = None,
        active_employee_ids:           Optional[set[str]]    = None,
    ) -> None:
        self._cooldown    = cooldown_seconds
        self._min_toggle  = min_toggle_seconds
        self._display_tz  = display_tz
        self._lock        = threading.RLock()

        if shift is not None:
            self._shift = shift
        else:
            is_overnight = (shift_end_hour, shift_end_minute) <= (shift_start_hour, shift_start_minute)
            self._shift = ShiftConfig(
                start_time=time(shift_start_hour, shift_start_minute),
                end_time=time(shift_end_hour, shift_end_minute),
                grace_period_minutes=late_threshold_minutes,
                early_leave_grace_minutes=early_leave_threshold_minutes,
                is_overnight=is_overnight,
            )

        # Nếu được cung cấp, chỉ chấp nhận nhân viên nằm trong tập active này
        self._active_employee_ids: Optional[set[str]] = (
            set(active_employee_ids) if active_employee_ids is not None else None
        )

        # Cooldown per (employee_id, camera_id) -> last scan epoch seconds
        self._last_scan: dict[tuple[str, str], float] = {}
        # Bản ghi chấm công theo ca: (employee_id, work_date) -> AttendanceRecordState
        self._records: dict[tuple[str, date], AttendanceRecordState] = {}

        self._events: list[AttendanceEvent] = []
        self._next_event_id: int = 1

    # ------------------------------------------------------------------
    # Quản lý trạng thái nhân viên active
    # ------------------------------------------------------------------

    def set_active_employees(self, employee_ids: Optional[set[str]]) -> None:
        """Cập nhật danh sách nhân viên đang hoạt động (None = không giới hạn)."""
        with self._lock:
            self._active_employee_ids = set(employee_ids) if employee_ids is not None else None

    # ------------------------------------------------------------------
    # API công khai chính
    # ------------------------------------------------------------------

    def process(
        self,
        match_result:        MatchResult,
        camera_id:           str                 = "cam_0",
        timestamp:           Optional[float]     = None,
        is_employee_active:  bool                = True,
        explicit_event_type: Optional[EventType] = None,
    ) -> AttendanceDecision:
        """
        Xử lý một kết quả nhận diện và quyết định tạo CHECK_IN / CHECK_OUT hay bỏ qua.
        Thread-safe: bảo vệ chống trùng lặp khi nhiều request gửi đồng thời.
        """
        now = timestamp if timestamp is not None else time_mod.time()

        # --- Bước 1: Chỉ xử lý KNOWN ---
        if match_result.status == MatchStatus.UNKNOWN:
            return AttendanceDecision(
                created=False,
                skip_reason=SkipReason.UNKNOWN,
                message="Người chưa đăng ký — không tạo sự kiện.",
            )
        if match_result.status == MatchStatus.UNCERTAIN:
            return AttendanceDecision(
                created=False,
                skip_reason=SkipReason.UNCERTAIN,
                message="Nhận diện không chắc chắn — cần đứng gần hoặc nhìn thẳng hơn.",
            )

        emp_id = match_result.employee_id  # non-None for KNOWN

        with self._lock:
            # --- Bước 2: Kiểm tra nhân viên còn active ---
            if not is_employee_active or (
                self._active_employee_ids is not None
                and emp_id not in self._active_employee_ids
            ):
                return AttendanceDecision(
                    created=False,
                    skip_reason=SkipReason.EMPLOYEE_INACTIVE,
                    message=f"Nhân viên {emp_id} đã bị vô hiệu hóa hoặc không tồn tại.",
                )

            # --- Bước 3: Chống nhận diện lặp liên tục (Cooldown) ---
            scan_key = (emp_id, camera_id)
            last_t = self._last_scan.get(scan_key, 0.0)
            elapsed = now - last_t
            if last_t > 0 and elapsed < self._cooldown:
                remaining = self._cooldown - elapsed
                return AttendanceDecision(
                    created=False,
                    skip_reason=SkipReason.COOLDOWN,
                    message=f"Cooldown: còn {remaining:.1f}s trước lần quét tiếp theo.",
                    cooldown_remaining=remaining,
                )

            # --- Bước 4: Xác định ca làm việc & khung giờ hợp lệ ---
            now_utc = datetime.fromtimestamp(now, tz=timezone.utc)
            now_local = now_utc.astimezone(self._display_tz)
            work_date, s_start_loc, s_end_loc, b_start_loc, b_end_loc = (
                self._shift.resolve_shift_window(now_local)
            )

            valid_from = s_start_loc - timedelta(minutes=self._shift.early_check_in_minutes)
            valid_until = s_end_loc + timedelta(minutes=self._shift.late_check_out_minutes)

            if now_local < valid_from or now_local > valid_until:
                return AttendanceDecision(
                    created=False,
                    skip_reason=SkipReason.OUTSIDE_SHIFT_HOURS,
                    message=(
                        f"Ngoài khung giờ ca {self._shift.shift_code} "
                        f"({valid_from.strftime('%H:%M')}–{valid_until.strftime('%H:%M')})."
                    ),
                )

            rec_key = (emp_id, work_date)
            rec = self._records.get(rec_key)

            # --- Bước 5: Quyết định CHECK_IN / CHECK_OUT dựa vào trạng thái ca ---
            if rec is None:
                # Chưa có bản ghi trong ca này
                checkout_only_threshold = s_end_loc - timedelta(
                    minutes=self._shift.missing_checkin_cutoff_minutes
                )
                is_missing_checkin_checkout = (
                    explicit_event_type == EventType.CHECK_OUT
                    or (explicit_event_type is None and now_local >= checkout_only_threshold)
                )

                if is_missing_checkin_checkout:
                    # Ngoại lệ: Không Check-in nhưng có Check-out -> INCOMPLETE (chờ HR đối soát)
                    early_mins = self._calc_early_leave_minutes(now_local, s_end_loc)
                    punctuality = (
                        PunctualityStatus.EARLY_LEAVE if early_mins > 0 else PunctualityStatus.UNKNOWN
                    )
                    rec = AttendanceRecordState(
                        employee_id=emp_id,
                        work_date=work_date,
                        shift_code=self._shift.shift_code,
                        shift_start_utc=s_start_loc.astimezone(timezone.utc),
                        shift_end_utc=s_end_loc.astimezone(timezone.utc),
                        break_start_utc=b_start_loc.astimezone(timezone.utc) if b_start_loc else None,
                        break_end_utc=b_end_loc.astimezone(timezone.utc) if b_end_loc else None,
                        attendance_status=AttendanceStatus.INCOMPLETE,
                        punctuality_status=punctuality,
                        check_in_time=None,
                        check_out_time=now_utc,
                        standard_work_minutes=self._shift.standard_work_minutes,
                        early_leave_minutes=early_mins,
                        needs_review=True,
                        review_reason="MISSING_CHECK_IN",
                    )
                    self._records[rec_key] = rec
                    event_type = EventType.CHECK_OUT
                else:
                    # Check-in bình thường -> IN_PROGRESS
                    punctuality, late_mins = self._calc_check_in_punctuality(now_local, s_start_loc)
                    rec = AttendanceRecordState(
                        employee_id=emp_id,
                        work_date=work_date,
                        shift_code=self._shift.shift_code,
                        shift_start_utc=s_start_loc.astimezone(timezone.utc),
                        shift_end_utc=s_end_loc.astimezone(timezone.utc),
                        break_start_utc=b_start_loc.astimezone(timezone.utc) if b_start_loc else None,
                        break_end_utc=b_end_loc.astimezone(timezone.utc) if b_end_loc else None,
                        attendance_status=AttendanceStatus.IN_PROGRESS,
                        punctuality_status=punctuality,
                        check_in_time=now_utc,
                        standard_work_minutes=self._shift.standard_work_minutes,
                        late_minutes=late_mins,
                    )
                    self._records[rec_key] = rec
                    event_type = EventType.CHECK_IN

            elif rec.attendance_status == AttendanceStatus.IN_PROGRESS:
                # Đã Check-in, chưa Check-out
                if explicit_event_type == EventType.CHECK_IN:
                    return AttendanceDecision(
                        created=False,
                        record=rec,
                        skip_reason=SkipReason.ALREADY_CHECKED_IN,
                        message=f"Nhân viên {emp_id} đã Check-in trong ca {work_date}.",
                    )

                # Kiểm tra khung giờ Check-out hợp lệ (tránh ra ngoài ăn trưa quay lại bị Check-out nhầm)
                auto_checkout_from = s_end_loc - timedelta(
                    minutes=self._shift.checkout_window_before_end_minutes
                )
                if explicit_event_type != EventType.CHECK_OUT and now_local < auto_checkout_from:
                    return AttendanceDecision(
                        created=False,
                        record=rec,
                        skip_reason=SkipReason.ALREADY_CHECKED_IN,
                        message=(
                            f"Đã Check-in lúc {rec.check_in_time.astimezone(self._display_tz).strftime('%H:%M')}. "
                            f"Chưa đến khung Check-out ({auto_checkout_from.strftime('%H:%M')}+)."
                        ),
                    )

                # Kiểm tra min_toggle nếu dùng explicit CHECK_OUT quá sát CHECK_IN
                if rec.check_in_time is not None:
                    since_ci = (now_utc - rec.check_in_time).total_seconds()
                    if since_ci < self._min_toggle:
                        rem = self._min_toggle - since_ci
                        return AttendanceDecision(
                            created=False,
                            record=rec,
                            skip_reason=SkipReason.MIN_TOGGLE,
                            message=f"Vừa CHECK_IN {since_ci:.0f}s trước — cần thêm {rem:.0f}s.",
                            cooldown_remaining=rem,
                        )

                # Ghi nhận CHECK_OUT -> COMPLETED và tính đầy đủ chỉ số thời gian
                self._apply_check_out(rec, now_utc, now_local, s_end_loc)
                event_type = EventType.CHECK_OUT

            elif rec.attendance_status == AttendanceStatus.COMPLETED:
                return AttendanceDecision(
                    created=False,
                    record=rec,
                    skip_reason=SkipReason.ALREADY_COMPLETED,
                    message=f"Nhân viên {emp_id} đã hoàn thành chấm công ca {work_date}.",
                )

            else:
                # INCOMPLETE (ví dụ đã có Check-out trước, nay bổ sung hoặc đã chốt)
                return AttendanceDecision(
                    created=False,
                    record=rec,
                    skip_reason=SkipReason.ALREADY_COMPLETED,
                    message=f"Bản ghi ca {work_date} ở trạng thái {rec.attendance_status.value} (chờ HR đối soát).",
                )

            # --- Bước 6: Tạo sự kiện & cập nhật cooldown ---
            event = AttendanceEvent(
                event_id=self._next_event_id,
                employee_id=emp_id,
                event_type=event_type,
                timestamp_utc=now,
                camera_id=camera_id,
                similarity=match_result.similarity,
                work_date=work_date,
                shift_code=self._shift.shift_code,
                attendance_status=rec.attendance_status,
                punctuality_status=rec.punctuality_status,
                late_minutes=rec.late_minutes if event_type == EventType.CHECK_IN else None,
                early_leave_minutes=rec.early_leave_minutes if event_type == EventType.CHECK_OUT else None,
                actual_work_minutes=rec.actual_work_minutes,
            )
            self._events.append(event)
            self._next_event_id += 1
            self._last_scan[scan_key] = now

            logger.info("AttendanceService: %s", event)
            return AttendanceDecision(
                created=True,
                event=event,
                record=rec,
                message=str(event),
            )

    # ------------------------------------------------------------------
    # Tính toán thời gian & trạng thái
    # ------------------------------------------------------------------

    def _calc_check_in_punctuality(
        self, checkin_local: datetime, shift_start_local: datetime
    ) -> tuple[PunctualityStatus, int]:
        grace_limit = shift_start_local + timedelta(minutes=self._shift.grace_period_minutes)
        if checkin_local <= grace_limit:
            return PunctualityStatus.ON_TIME, 0
        late_mins = max(1, int((checkin_local - shift_start_local).total_seconds() // 60))
        return PunctualityStatus.LATE, late_mins

    def _calc_early_leave_minutes(
        self, checkout_local: datetime, shift_end_local: datetime
    ) -> int:
        early_limit = shift_end_local - timedelta(minutes=self._shift.early_leave_grace_minutes)
        if checkout_local >= early_limit:
            return 0
        return max(1, int((shift_end_local - checkout_local).total_seconds() // 60))

    def _calc_break_overlap_minutes(
        self,
        check_in_utc: datetime,
        check_out_utc: datetime,
        break_start_utc: Optional[datetime],
        break_end_utc: Optional[datetime],
    ) -> int:
        """Chỉ trừ thời gian nghỉ thực tế giao với khoảng [check_in, check_out]."""
        if break_start_utc is not None and break_end_utc is not None:
            overlap_start = max(check_in_utc, break_start_utc)
            overlap_end = min(check_out_utc, break_end_utc)
            if overlap_end > overlap_start:
                return int((overlap_end - overlap_start).total_seconds() // 60)
            return 0
        return 0

    def _apply_check_out(
        self,
        rec: AttendanceRecordState,
        checkout_utc: datetime,
        checkout_local: datetime,
        shift_end_local: datetime,
    ) -> None:
        rec.check_out_time = checkout_utc
        rec.attendance_status = AttendanceStatus.COMPLETED

        if rec.check_in_time is not None:
            presence_mins = max(0, int((checkout_utc - rec.check_in_time).total_seconds() // 60))
            break_mins = self._calc_break_overlap_minutes(
                rec.check_in_time, checkout_utc, rec.break_start_utc, rec.break_end_utc
            )
            actual_mins = max(0, presence_mins - break_mins)
            std_mins = rec.standard_work_minutes

            rec.total_presence_minutes = presence_mins
            rec.break_minutes = break_mins
            rec.actual_work_minutes = actual_mins
            rec.missing_minutes = max(0, std_mins - actual_mins)
            rec.extra_minutes = max(0, actual_mins - std_mins)

        early_mins = self._calc_early_leave_minutes(checkout_local, shift_end_local)
        rec.early_leave_minutes = early_mins
        if early_mins > 0:
            if rec.punctuality_status == PunctualityStatus.LATE:
                rec.punctuality_status = PunctualityStatus.LATE_AND_EARLY
            elif rec.punctuality_status in (PunctualityStatus.ON_TIME, PunctualityStatus.UNKNOWN):
                rec.punctuality_status = PunctualityStatus.EARLY_LEAVE

    # ------------------------------------------------------------------
    # Đối soát cuối ca & Phê duyệt tăng ca
    # ------------------------------------------------------------------

    def reconcile_shift(
        self,
        work_date: date,
        now_timestamp: Optional[float] = None,
    ) -> list[AttendanceRecordState]:
        """
        Đối soát cuối ca cho ngày `work_date`.
        Chỉ chuyển các bản ghi IN_PROGRESS sang INCOMPLETE sau khi ca kết thúc
        (bao gồm cả late_check_out_minutes). Không kết luận nghỉ không phép.
        """
        now_ts = now_timestamp if now_timestamp is not None else time_mod.time()
        now_utc = datetime.fromtimestamp(now_ts, tz=timezone.utc)
        reconciled: list[AttendanceRecordState] = []

        with self._lock:
            for (emp_id, rec_date), rec in self._records.items():
                if rec_date != work_date:
                    continue
                if rec.attendance_status != AttendanceStatus.IN_PROGRESS:
                    continue
                cutoff_utc = rec.shift_end_utc + timedelta(
                    minutes=self._shift.late_check_out_minutes
                )
                if now_utc <= cutoff_utc:
                    continue
                rec.attendance_status = AttendanceStatus.INCOMPLETE
                rec.needs_review = True
                rec.review_reason = "MISSING_CHECK_OUT"
                reconciled.append(rec)

        return reconciled

    def approve_overtime(
        self,
        employee_id: str,
        work_date: date,
        approved_minutes: int,
    ) -> AttendanceRecordState:
        """Phê duyệt tăng ca hợp lệ (OvertimeMinutes), tách biệt với ExtraMinutes."""
        if approved_minutes < 0:
            raise ValueError("approved_minutes phải >= 0")
        with self._lock:
            rec = self._records.get((employee_id, work_date))
            if rec is None:
                raise ValueError(f"Không tìm thấy bản ghi {employee_id} ngày {work_date}")
            rec.overtime_minutes = approved_minutes
            rec.overtime_approved = approved_minutes > 0
            return rec

    def get_record(self, employee_id: str, work_date: date) -> Optional[AttendanceRecordState]:
        with self._lock:
            return self._records.get((employee_id, work_date))

    # ------------------------------------------------------------------
    # Truy vấn lịch sử
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[AttendanceEvent]:
        with self._lock:
            return list(self._events)

    def events_for(
        self,
        employee_id: Optional[str] = None,
        camera_id:   Optional[str] = None,
        event_type:  Optional[EventType] = None,
    ) -> list[AttendanceEvent]:
        with self._lock:
            result = list(self._events)
        if employee_id is not None:
            result = [e for e in result if e.employee_id == employee_id]
        if camera_id is not None:
            result = [e for e in result if e.camera_id == camera_id]
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        return result

    def last_event_for(
        self,
        employee_id: str,
        camera_id:   Optional[str] = None,
    ) -> Optional[AttendanceEvent]:
        events = self.events_for(employee_id=employee_id, camera_id=camera_id)
        return events[-1] if events else None

    @property
    def total_events(self) -> int:
        with self._lock:
            return len(self._events)

    def cooldown_remaining(self, employee_id: str, camera_id: str = "cam_0") -> float:
        with self._lock:
            last_t = self._last_scan.get((employee_id, camera_id), 0.0)
        if last_t == 0.0:
            return 0.0
        elapsed = time_mod.time() - last_t
        return max(0.0, self._cooldown - elapsed)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_state(
        self,
        employee_id: Optional[str] = None,
        camera_id:   Optional[str] = None,
    ) -> None:
        with self._lock:
            if employee_id is None and camera_id is None:
                self._last_scan.clear()
                self._records.clear()
            else:
                scan_keys = [
                    k for k in self._last_scan
                    if (employee_id is None or k[0] == employee_id)
                    and (camera_id is None or k[1] == camera_id)
                ]
                for k in scan_keys:
                    del self._last_scan[k]
                if camera_id is None:
                    rec_keys = [k for k in self._records if k[0] == employee_id]
                    for k in rec_keys:
                        del self._records[k]

    def clear_history(self) -> None:
        with self._lock:
            self._events.clear()
            self._next_event_id = 1

    # ------------------------------------------------------------------
    # Export & Lưu trữ phân cấp theo Năm / Tháng / Ngày
    # ------------------------------------------------------------------

    @staticmethod
    def get_daily_dir(base_dir: str | Path, target_date: date) -> Path:
        return (
            Path(base_dir)
            / f"{target_date.year:04d}"
            / f"{target_date.month:02d}"
            / f"{target_date.day:02d}"
        )

    def export_json(self, filepath: str, events: Optional[list[AttendanceEvent]] = None) -> None:
        target_events = self.events if events is None else events
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in target_events], f, ensure_ascii=False, indent=2)

    def export_csv(self, filepath: str, events: Optional[list[AttendanceEvent]] = None) -> None:
        target_events = self.events if events is None else events
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "event_id", "employee_id", "event_type", "work_date", "shift_code",
            "timestamp_local", "timestamp_utc", "camera_id", "similarity",
            "attendance_status", "punctuality_status", "late_minutes",
            "early_leave_minutes", "actual_work_minutes",
            "verify_method", "verify_status", "result",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in target_events:
                row = e.to_dict()
                writer.writerow({k: row[k] for k in fieldnames})

    def export_by_date(
        self,
        base_dir: str | Path,
        tz: Optional[timezone] = None,
        target_date: Optional[date] = None,
    ) -> list[Path]:
        use_tz = tz or self._display_tz
        grouped: dict[date, list[AttendanceEvent]] = {}
        for ev in self.events:
            ev_date = ev.work_date or ev.to_local_dt(use_tz).date()
            grouped.setdefault(ev_date, []).append(ev)

        if not grouped:
            fallback_date = target_date or datetime.now(use_tz).date()
            grouped[fallback_date] = []

        written_dirs: list[Path] = []
        for day in sorted(grouped):
            day_dir = self.get_daily_dir(base_dir, day)
            day_dir.mkdir(parents=True, exist_ok=True)
            day_events = grouped[day]
            self.export_json(str(day_dir / "attendance.json"), events=day_events)
            self.export_csv(str(day_dir / "attendance.csv"), events=day_events)
            written_dirs.append(day_dir)

        return written_dirs

    def print_summary(self) -> None:
        print(f"\n{'='*68}")
        print(f"  LỊCH SỬ CHẤM CÔNG  ({self.total_events} sự kiện)")
        print(f"{'='*68}")
        evts = self.events
        if not evts:
            print("  (Chưa có sự kiện nào)")
        for ev in evts:
            icon = "→" if ev.event_type == EventType.CHECK_IN else "←"
            dt = ev.to_local_dt(self._display_tz)
            print(
                f"  {icon}  [{ev.event_id:03d}] {ev.employee_id:<14} "
                f"{ev.event_type.value:<10} {dt.strftime('%H:%M:%S')}  "
                f"{ev.attendance_status.value:<11} sim={ev.similarity:.3f}  cam={ev.camera_id}"
            )
        print(f"{'='*68}\n")
