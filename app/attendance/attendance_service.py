"""
app/attendance/attendance_service.py
Logic chấm công — Giai đoạn 8.

Chuyển kết quả nhận diện khuôn mặt (MatchResult) thành sự kiện chấm công
(AttendanceEvent) theo quy tắc nghiệp vụ.

Quy tắc cốt lõi:
  1. Cooldown chống ghi lặp:
       Mỗi nhân viên phải cách ít nhất COOLDOWN_SECONDS giây giữa hai sự kiện
       liên tiếp trên cùng một camera. Người đứng liên tục trước camera KHÔNG
       tạo hàng loạt bản ghi.

  2. Luân phiên CHECK_IN / CHECK_OUT:
       Lần nhận diện hợp lệ đầu tiên trong ngày → CHECK_IN.
       Lần hợp lệ tiếp theo sau ít nhất MIN_TOGGLE_SECONDS → CHECK_OUT.
       Tiếp tục luân phiên IN/OUT.
       (Logic này sẽ được tinh chỉnh theo ca làm việc ở Giai đoạn 9+.)

  3. Chỉ xử lý kết quả KNOWN:
       UNCERTAIN và UNKNOWN không tạo sự kiện; thay vào đó trả về reason
       rõ ràng để UI hiển thị hướng dẫn.

  4. Timestamp UTC nội bộ, múi giờ chỉ dùng khi hiển thị:
       Toàn bộ timestamp lưu dưới dạng float (epoch seconds, UTC).
       Helper to_local_dt() chuyển sang múi giờ cấu hình.

Thiết kế:
  - Không phụ thuộc camera hay UI; nhận MatchResult + timestamp.
  - Lịch sử sự kiện hiện tại lưu in-memory (list); Giai đoạn 9 thay bằng DB.
  - Có thể export lịch sử ra file JSON/CSV để kiểm tra khi chưa có DB.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

from app.vision.face_matcher import MatchResult, MatchStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hằng số / ngưỡng mặc định
# ---------------------------------------------------------------------------

# Thời gian chờ tối thiểu (giây) giữa hai sự kiện của cùng một người trên cùng camera
DEFAULT_COOLDOWN_SECONDS: float = 30.0

# Thời gian tối thiểu (giây) giữa CHECK_IN và CHECK_OUT đầu tiên
# (tránh toggle quá nhanh do vô tình đứng trước camera hai lần)
DEFAULT_MIN_TOGGLE_SECONDS: float = 60.0

# Múi giờ hiển thị mặc định (UTC+7 — Hà Nội / Tp.HCM)
DEFAULT_TIMEZONE = timezone(timedelta(hours=7))


# ---------------------------------------------------------------------------
# Enum kiểu sự kiện chấm công
# ---------------------------------------------------------------------------

class EventType(Enum):
    CHECK_IN  = "check_in"   # Lượt vào (lần nhận diện hợp lệ đầu)
    CHECK_OUT = "check_out"  # Lượt ra (lần nhận diện hợp lệ tiếp theo)


# ---------------------------------------------------------------------------
# Enum lý do KHÔNG tạo sự kiện
# ---------------------------------------------------------------------------

class SkipReason(Enum):
    COOLDOWN        = "cooldown"         # Chưa hết thời gian chờ
    UNCERTAIN       = "uncertain"        # Nhận diện không chắc chắn
    UNKNOWN         = "unknown"          # Người chưa đăng ký
    MIN_TOGGLE      = "min_toggle"       # Chưa đủ thời gian để đảo chiều IN↔OUT


# ---------------------------------------------------------------------------
# Sự kiện chấm công
# ---------------------------------------------------------------------------

@dataclass
class AttendanceEvent:
    """
    Một sự kiện chấm công (check-in hoặc check-out).

    Thuộc tính:
        event_id      : ID tăng dần trong phiên (thay bằng UUID khi có DB)
        employee_id   : mã nhân viên
        event_type    : CHECK_IN hoặc CHECK_OUT
        timestamp_utc : thời điểm sự kiện (epoch seconds, UTC)
        camera_id     : ID camera ghi nhận (chuỗi tự do)
        similarity    : cosine similarity tại thời điểm nhận diện
        confidence    : độ tin cậy sự kiện (thường = similarity)
    """
    event_id:      int
    employee_id:   str
    event_type:    EventType
    timestamp_utc: float
    camera_id:     str   = "cam_0"
    similarity:    float = 0.0

    def to_local_dt(self, tz: timezone = DEFAULT_TIMEZONE) -> datetime:
        """Chuyển timestamp UTC sang datetime theo múi giờ cấu hình."""
        return datetime.fromtimestamp(self.timestamp_utc, tz=tz)

    def __str__(self) -> str:
        dt = self.to_local_dt()
        return (
            f"[{self.event_type.value.upper()}] "
            f"{self.employee_id} | "
            f"{dt.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"sim={self.similarity:.3f} | "
            f"cam={self.camera_id}"
        )

    def to_dict(self) -> dict:
        return {
            "event_id":      self.event_id,
            "employee_id":   self.employee_id,
            "event_type":    self.event_type.value,
            "timestamp_utc": self.timestamp_utc,
            "timestamp_local": self.to_local_dt().isoformat(),
            "camera_id":     self.camera_id,
            "similarity":    round(self.similarity, 4),
        }


# ---------------------------------------------------------------------------
# Kết quả xử lý một lần nhận diện
# ---------------------------------------------------------------------------

@dataclass
class AttendanceDecision:
    """
    Kết quả quyết định cho một lần nhận diện.

    Thuộc tính:
        created       : True → đã tạo sự kiện chấm công
        event         : AttendanceEvent (None nếu không tạo)
        skip_reason   : lý do bỏ qua (None nếu đã tạo)
        message       : thông điệp hiển thị cho operator / UI
        cooldown_remaining : giây còn lại trong thời gian chờ (nếu áp dụng)
    """
    created:            bool
    event:              Optional[AttendanceEvent] = None
    skip_reason:        Optional[SkipReason]      = None
    message:            str                       = ""
    cooldown_remaining: float                     = 0.0

    @property
    def employee_id(self) -> Optional[str]:
        return self.event.employee_id if self.event else None


# ---------------------------------------------------------------------------
# Trạng thái theo dõi từng nhân viên theo camera
# ---------------------------------------------------------------------------

@dataclass
class _EmployeeState:
    """Trạng thái nội bộ theo dõi một nhân viên trên một camera."""
    last_event_time: float       = 0.0    # Epoch seconds (UTC) lần sự kiện gần nhất
    last_event_type: Optional[EventType] = None


# ---------------------------------------------------------------------------
# Dịch vụ chấm công
# ---------------------------------------------------------------------------

class AttendanceService:
    """
    Áp dụng quy tắc nghiệp vụ chấm công vào kết quả nhận diện khuôn mặt.

    Tham số khởi tạo:
        cooldown_seconds    : Giây chờ tối thiểu giữa hai sự kiện cùng người/camera
        min_toggle_seconds  : Giây tối thiểu để chuyển từ CHECK_IN → CHECK_OUT
        display_tz          : Múi giờ dùng cho hiển thị (không ảnh hưởng lưu trữ)

    Ví dụ sử dụng::

        service = AttendanceService()
        decision = service.process(match_result, camera_id="cam_main")

        if decision.created:
            print(decision.event)      # [CHECK_IN] EMP001 | 2026-11-13 08:05:12 ...
        else:
            print(decision.message)    # Cooldown: còn 25.3 giây
    """

    def __init__(
        self,
        cooldown_seconds:   float    = DEFAULT_COOLDOWN_SECONDS,
        min_toggle_seconds: float    = DEFAULT_MIN_TOGGLE_SECONDS,
        display_tz:         timezone = DEFAULT_TIMEZONE,
    ) -> None:
        self._cooldown    = cooldown_seconds
        self._min_toggle  = min_toggle_seconds
        self._display_tz  = display_tz

        # Trạng thái theo dõi: (employee_id, camera_id) → _EmployeeState
        self._states: dict[tuple[str, str], _EmployeeState] = {}

        # Lịch sử sự kiện (thay bằng DB ở Giai đoạn 9)
        self._events: list[AttendanceEvent] = []
        self._next_event_id: int = 1

    # ------------------------------------------------------------------
    # API công khai chính
    # ------------------------------------------------------------------

    def process(
        self,
        match_result: MatchResult,
        camera_id:    str   = "cam_0",
        timestamp:    Optional[float] = None,
    ) -> AttendanceDecision:
        """
        Xử lý một kết quả nhận diện và quyết định có tạo sự kiện hay không.

        Tham số:
            match_result : MatchResult từ FaceMatcher
            camera_id    : ID camera ghi nhận
            timestamp    : Epoch seconds UTC (None → dùng thời gian hiện tại)

        Trả về:
            AttendanceDecision — xem mô tả dataclass bên trên
        """
        now = timestamp if timestamp is not None else time.time()

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

        emp_id = match_result.employee_id  # str, guaranteed non-None for KNOWN

        # --- Bước 2: Kiểm tra cooldown ---
        state_key = (emp_id, camera_id)
        state = self._states.setdefault(state_key, _EmployeeState())
        elapsed = now - state.last_event_time

        if state.last_event_time > 0 and elapsed < self._cooldown:
            remaining = self._cooldown - elapsed
            return AttendanceDecision(
                created=False,
                skip_reason=SkipReason.COOLDOWN,
                message=f"Cooldown: còn {remaining:.1f}s trước sự kiện tiếp theo.",
                cooldown_remaining=remaining,
            )

        # --- Bước 3: Xác định loại sự kiện (CHECK_IN / CHECK_OUT) ---
        if state.last_event_type is None:
            # Lần đầu trong ngày (hoặc phiên) → CHECK_IN
            event_type = EventType.CHECK_IN
        else:
            # Đảo chiều: IN↔OUT, nhưng kiểm tra thêm min_toggle
            next_type = (
                EventType.CHECK_OUT
                if state.last_event_type == EventType.CHECK_IN
                else EventType.CHECK_IN
            )
            # CHECK_IN → CHECK_OUT phải đủ thời gian (tránh toggle giả)
            if (
                state.last_event_type == EventType.CHECK_IN
                and next_type == EventType.CHECK_OUT
                and elapsed < self._min_toggle
            ):
                remaining = self._min_toggle - elapsed
                return AttendanceDecision(
                    created=False,
                    skip_reason=SkipReason.MIN_TOGGLE,
                    message=(
                        f"Vừa CHECK_IN {elapsed:.0f}s trước — "
                        f"cần thêm {remaining:.0f}s để ghi CHECK_OUT."
                    ),
                    cooldown_remaining=remaining,
                )
            event_type = next_type

        # --- Bước 4: Tạo sự kiện ---
        event = AttendanceEvent(
            event_id=self._next_event_id,
            employee_id=emp_id,
            event_type=event_type,
            timestamp_utc=now,
            camera_id=camera_id,
            similarity=match_result.similarity,
        )
        self._events.append(event)
        self._next_event_id += 1

        # Cập nhật trạng thái
        state.last_event_time = now
        state.last_event_type = event_type

        logger.info("AttendanceService: %s", event)
        return AttendanceDecision(
            created=True,
            event=event,
            message=str(event),
        )

    # ------------------------------------------------------------------
    # Truy vấn lịch sử
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[AttendanceEvent]:
        """Toàn bộ lịch sử sự kiện trong phiên hiện tại (bản sao)."""
        return list(self._events)

    def events_for(
        self,
        employee_id: Optional[str] = None,
        camera_id:   Optional[str] = None,
        event_type:  Optional[EventType] = None,
    ) -> list[AttendanceEvent]:
        """
        Lọc sự kiện theo tiêu chí.

        Tất cả tham số là tùy chọn; không truyền = không lọc theo tiêu chí đó.
        """
        result = self._events
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
        """Sự kiện gần nhất của nhân viên (tùy chọn lọc theo camera)."""
        events = self.events_for(employee_id=employee_id, camera_id=camera_id)
        return events[-1] if events else None

    @property
    def total_events(self) -> int:
        return len(self._events)

    def cooldown_remaining(self, employee_id: str, camera_id: str = "cam_0") -> float:
        """
        Trả về số giây còn lại trong cooldown của nhân viên trên camera.
        0.0 nếu đã hết hoặc chưa có sự kiện nào.
        """
        state = self._states.get((employee_id, camera_id))
        if state is None or state.last_event_time == 0.0:
            return 0.0
        elapsed = time.time() - state.last_event_time
        return max(0.0, self._cooldown - elapsed)

    # ------------------------------------------------------------------
    # Reset (hữu ích khi bắt đầu ngày mới hoặc ca mới)
    # ------------------------------------------------------------------

    def reset_state(
        self,
        employee_id: Optional[str] = None,
        camera_id:   Optional[str] = None,
    ) -> None:
        """
        Xóa trạng thái cooldown/tracking.

        - reset_state()                           : Xóa tất cả
        - reset_state("EMP001")                   : Chỉ EMP001 trên mọi camera
        - reset_state("EMP001", camera_id="cam1") : Chỉ EMP001 trên cam1
        """
        if employee_id is None and camera_id is None:
            self._states.clear()
            logger.info("AttendanceService.reset_state: đã xóa tất cả.")
        else:
            keys_to_del = [
                k for k in self._states
                if (employee_id is None or k[0] == employee_id)
                and (camera_id is None or k[1] == camera_id)
            ]
            for k in keys_to_del:
                del self._states[k]
            logger.info(
                "AttendanceService.reset_state: đã xóa %d trạng thái.", len(keys_to_del)
            )

    def clear_history(self) -> None:
        """Xóa toàn bộ lịch sử sự kiện (không ảnh hưởng trạng thái cooldown)."""
        self._events.clear()
        self._next_event_id = 1
        logger.info("AttendanceService.clear_history: đã xóa lịch sử.")

    # ------------------------------------------------------------------
    # Export (tạm thời trước khi có DB)
    # ------------------------------------------------------------------

    def export_json(self, filepath: str) -> None:
        """
        Xuất toàn bộ lịch sử sự kiện ra file JSON.
        Dùng để kiểm tra khi chưa có Database.
        """
        import pathlib
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._events], f, ensure_ascii=False, indent=2)
        logger.info("AttendanceService.export_json: %d sự kiện → %s", len(self._events), filepath)

    def export_csv(self, filepath: str) -> None:
        """
        Xuất toàn bộ lịch sử sự kiện ra file CSV.
        """
        import pathlib
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["event_id", "employee_id", "event_type",
                      "timestamp_local", "timestamp_utc", "camera_id", "similarity"]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self._events:
                row = e.to_dict()
                writer.writerow({k: row[k] for k in fieldnames})
        logger.info("AttendanceService.export_csv: %d sự kiện → %s", len(self._events), filepath)

    def print_summary(self) -> None:
        """In tóm tắt lịch sử ra console (debug / demo)."""
        print(f"\n{'='*58}")
        print(f"  LỊCH SỬ CHẤM CÔNG  ({self.total_events} sự kiện)")
        print(f"{'='*58}")
        if not self._events:
            print("  (Chưa có sự kiện nào)")
        for ev in self._events:
            icon = "→" if ev.event_type == EventType.CHECK_IN else "←"
            dt   = ev.to_local_dt(self._display_tz)
            print(
                f"  {icon}  [{ev.event_id:03d}] {ev.employee_id:<12} "
                f"{ev.event_type.value:<10}  "
                f"{dt.strftime('%H:%M:%S')}  "
                f"sim={ev.similarity:.3f}  "
                f"cam={ev.camera_id}"
            )
        print(f"{'='*58}\n")

