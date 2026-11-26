"""
app/gui/main_window.py
Giao diện GUI chính cho FaceAttend — dùng Tkinter + OpenCV.

Các chức năng:
  Tab 1 — Điểm Danh:
    • Hiển thị video camera real-time với bounding box nhận diện
    • Hiển thị "Mã - Tên" nhân viên khi nhận diện thành công
    • Bảng lịch sử chấm công trong phiên (10 dòng gần nhất)
    • Nút: Bắt đầu / Dừng camera, Reset cooldown, Xuất log

  Tab 2 — Đăng Ký Nhân Viên:
    • Form nhập Tên, Phòng ban; Mã NV tự động sinh
    • Preview camera + nút Chụp mẫu, Progress bar
    • Nút Finalize (lưu gallery)
    • Danh sách nhân viên đã đăng ký trong gallery

  Tab 3 — Gallery:
    • Danh sách nhân viên đã đăng ký (tên, mã, số mẫu)
    • Nút xóa nhân viên khỏi gallery

Yêu cầu:
    pip install pillow  # PIL để convert frame OpenCV → Tkinter ImageTk
"""

from __future__ import annotations

import logging
import os
import sys
import time
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.camera.camera_capture import CameraCapture
from app.vision.face_detector import FaceDetector
from app.vision.face_alignment import align_face, crop_face_simple, ALIGNED_SIZE
from app.vision.face_quality import FaceQualityChecker, QualityThresholds
from app.vision.face_recognizer import FaceRecognizer
from app.vision.face_matcher import FaceGallery, FaceMatcher, MatchStatus, MatchThresholds
from app.attendance.attendance_service import (
    AttendanceService, EventType, PunctualityStatus, SkipReason, DEFAULT_TIMEZONE,
)
from app.enrollment.enrollment_service import (
    EnrollmentService, SampleStatus, MIN_SAMPLES_REQUIRED,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
DETECTOR_PATH   = ROOT / "models" / "det_10g.onnx"
RECOGNIZER_PATH = ROOT / "models" / "w600k_r50.onnx"
GALLERY_FILE    = ROOT / "data" / "gallery.npz"
ATTENDANCE_DIR  = ROOT / "data" / "attendance"
FACES_DIR       = ROOT / "data" / "registered_faces"

COMPANY_CODE = os.environ.get("COMPANY_CODE", "FA")

# ---------------------------------------------------------------------------
# Màu sắc HUD (BGR cho OpenCV, RGB cho Tkinter)
# ---------------------------------------------------------------------------
CV_GREEN  = (0, 220, 0)
CV_RED    = (0, 0, 220)
CV_YELLOW = (0, 200, 220)
CV_CYAN   = (220, 220, 0)
CV_WHITE  = (220, 220, 220)
CV_ORANGE = (0, 140, 255)
CV_GRAY   = (140, 140, 140)
FONT      = cv2.FONT_HERSHEY_SIMPLEX

TK_BG        = "#1e1e2e"
TK_PANEL     = "#2a2a3e"
TK_ACCENT    = "#7c3aed"
TK_GREEN     = "#22c55e"
TK_RED       = "#ef4444"
TK_YELLOW    = "#eab308"
TK_TEXT      = "#e2e8f0"
TK_SUBTEXT   = "#94a3b8"


def _cv_put(img, text, pos, color=CV_WHITE, scale=0.5, thick=1):
    cv2.putText(img, text, pos, FONT, scale, color, thick, cv2.LINE_AA)


def _status_color(status: MatchStatus):
    if status == MatchStatus.KNOWN:     return CV_GREEN
    if status == MatchStatus.UNCERTAIN: return CV_YELLOW
    return CV_RED


def _now_local() -> str:
    return datetime.now(DEFAULT_TIMEZONE).strftime("%H:%M:%S %d/%m/%Y")


# ---------------------------------------------------------------------------
# Lớp quản lý AI components (singleton, load một lần)
# ---------------------------------------------------------------------------

class AIEngine:
    """Khởi tạo và giữ các AI component dùng chung giữa các tab."""

    def __init__(self) -> None:
        self.detector:   Optional[FaceDetector]      = None
        self.recognizer: Optional[FaceRecognizer]    = None
        self.checker:    Optional[FaceQualityChecker] = None
        self.gallery:    FaceGallery                 = FaceGallery()
        self.matcher:    Optional[FaceMatcher]       = None
        self.ready:      bool                        = False
        self._error:     str                         = ""

    def load(self, progress_cb=None) -> bool:
        """Tải model AI. Gọi trong thread riêng để không block UI."""
        try:
            if progress_cb:
                progress_cb("Đang tải SCRFD detector...")
            self.detector = FaceDetector(
                DETECTOR_PATH, conf_threshold=0.5, nms_threshold=0.3, min_face_size=40
            )

            if progress_cb:
                progress_cb("Đang tải ArcFace recognizer...")
            self.recognizer = FaceRecognizer(RECOGNIZER_PATH)

            self.checker = FaceQualityChecker(QualityThresholds(
                min_face_size_px=50,
                min_sharpness=35.0,
                min_brightness=25.0,
                max_brightness=235.0,
                max_yaw_deg=35.0,
                min_confidence=0.45,
            ))

            if progress_cb:
                progress_cb("Đang nạp gallery nhân viên...")
            self.gallery = FaceGallery.load(GALLERY_FILE)

            thresholds = MatchThresholds(accept=0.35, reject=0.20, margin_min=0.07)
            self.matcher = FaceMatcher(self.gallery, thresholds)

            self.ready = True
            if progress_cb:
                progress_cb(f"Sẵn sàng! Gallery: {self.gallery.size} nhân viên.")
            return True

        except Exception as exc:
            self._error = str(exc)
            if progress_cb:
                progress_cb(f"LỖI: {exc}")
            logger.error("AIEngine.load lỗi: %s", exc, exc_info=True)
            return False

    def reload_gallery(self) -> None:
        """Nạp lại gallery sau khi đăng ký nhân viên mới."""
        self.gallery = FaceGallery.load(GALLERY_FILE)
        if self.matcher:
            self.matcher.gallery = self.gallery


# ---------------------------------------------------------------------------
# Tab 1: Điểm Danh
# ---------------------------------------------------------------------------

class AttendanceTab(ttk.Frame):
    """Tab điểm danh: stream camera + nhận diện + lịch sử."""

    COOLDOWN_SECONDS   = 30.0
    MIN_TOGGLE_SECONDS = 60.0
    CAMERA_ID          = "cam_0"

    def __init__(self, parent: ttk.Notebook, engine: AIEngine) -> None:
        super().__init__(parent)
        self._engine      = engine
        self._cam:  Optional[CameraCapture] = None
        self._running     = False
        self._thread:     Optional[threading.Thread] = None
        self._attendance  = AttendanceService(
            cooldown_seconds=self.COOLDOWN_SECONDS,
            min_toggle_seconds=self.MIN_TOGGLE_SECONDS,
        )
        self._recent_events: list[str] = []
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # --- Camera panel ---
        cam_frame = ttk.LabelFrame(self, text=" 📷  Camera nhận diện ", padding=4)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        cam_frame.rowconfigure(0, weight=1)
        cam_frame.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(cam_frame, bg="#000", width=640, height=480)
        self._canvas.grid(row=0, column=0, sticky="nsew")

        # Nút điều khiển camera
        btn_frame = ttk.Frame(cam_frame)
        btn_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self._btn_start = ttk.Button(btn_frame, text="▶  Bắt đầu", command=self._start)
        self._btn_start.pack(side="left", padx=4)

        self._btn_stop = ttk.Button(btn_frame, text="⏹  Dừng", command=self._stop, state="disabled")
        self._btn_stop.pack(side="left", padx=4)

        ttk.Button(btn_frame, text="↺  Reset cooldown", command=self._reset).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="💾  Xuất log", command=self._export).pack(side="right", padx=4)

        self._status_var = tk.StringVar(value="⏸ Camera chưa bật")
        ttk.Label(cam_frame, textvariable=self._status_var, foreground=TK_SUBTEXT).grid(
            row=2, column=0, sticky="w", pady=(2, 0)
        )

        # --- Side panel: thống kê + lịch sử ---
        side = ttk.Frame(self)
        side.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        side.rowconfigure(2, weight=1)
        side.columnconfigure(0, weight=1)

        # Thẻ thông tin
        info_frame = ttk.LabelFrame(side, text=" 📊  Thống kê phiên ", padding=8)
        info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        info_frame.columnconfigure(1, weight=1)

        self._var_total  = tk.StringVar(value="0")
        self._var_in     = tk.StringVar(value="0")
        self._var_out    = tk.StringVar(value="0")
        self._var_gallery = tk.StringVar(value="0")

        for i, (lbl, var) in enumerate([
            ("Tổng sự kiện:", self._var_total),
            ("Check-In:", self._var_in),
            ("Check-Out:", self._var_out),
            ("Gallery (NV):", self._var_gallery),
        ]):
            ttk.Label(info_frame, text=lbl, foreground=TK_SUBTEXT).grid(row=i, column=0, sticky="w")
            ttk.Label(info_frame, textvariable=var, font=("Arial", 11, "bold")).grid(
                row=i, column=1, sticky="e"
            )

        # Card nhận diện gần nhất
        recog_frame = ttk.LabelFrame(side, text=" 🎯  Nhận diện gần nhất ", padding=8)
        recog_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        self._var_last_id   = tk.StringVar(value="—")
        self._var_last_name = tk.StringVar(value="—")
        self._var_last_time = tk.StringVar(value="—")
        self._var_last_type = tk.StringVar(value="—")
        self._var_last_punctuality = tk.StringVar(value="—")
        self._var_last_device = tk.StringVar(value="—")
        self._var_last_method = tk.StringVar(value="Face Recognition")
        self._var_last_vstatus = tk.StringVar(value="—")
        self._var_last_result = tk.StringVar(value="—")
        self._var_last_record_id = tk.StringVar(value="—")

        ttk.Label(recog_frame, text="Mã NV:", foreground=TK_SUBTEXT).grid(row=0, column=0, sticky="w")
        ttk.Label(recog_frame, textvariable=self._var_last_id,
                  font=("Consolas", 11, "bold"), foreground=TK_ACCENT).grid(row=0, column=1, sticky="w")

        ttk.Label(recog_frame, text="Tên:", foreground=TK_SUBTEXT).grid(row=1, column=0, sticky="w")
        ttk.Label(recog_frame, textvariable=self._var_last_name,
                  font=("Arial", 11, "bold")).grid(row=1, column=1, sticky="w")

        ttk.Label(recog_frame, text="Loại:", foreground=TK_SUBTEXT).grid(row=2, column=0, sticky="w")
        ttk.Label(recog_frame, textvariable=self._var_last_type,
                  font=("Arial", 11, "bold")).grid(row=2, column=1, sticky="w")

        ttk.Label(recog_frame, text="Thời gian:", foreground=TK_SUBTEXT).grid(row=3, column=0, sticky="w")
        ttk.Label(recog_frame, textvariable=self._var_last_time).grid(row=3, column=1, sticky="w")

        ttk.Label(recog_frame, text="Trạng thái:", foreground=TK_SUBTEXT).grid(row=4, column=0, sticky="w")
        ttk.Label(recog_frame, textvariable=self._var_last_punctuality,
                  font=("Arial", 10, "bold")).grid(row=4, column=1, sticky="w")

        ttk.Label(recog_frame, text="Device ID:", foreground=TK_SUBTEXT).grid(row=5, column=0, sticky="w")
        ttk.Label(recog_frame, textvariable=self._var_last_device).grid(row=5, column=1, sticky="w")

        ttk.Label(recog_frame, text="Phương thức:", foreground=TK_SUBTEXT).grid(row=6, column=0, sticky="w")
        ttk.Label(recog_frame, textvariable=self._var_last_method).grid(row=6, column=1, sticky="w")

        ttk.Label(recog_frame, text="Verify Status:", foreground=TK_SUBTEXT).grid(row=7, column=0, sticky="w")
        ttk.Label(recog_frame, textvariable=self._var_last_vstatus).grid(row=7, column=1, sticky="w")

        ttk.Label(recog_frame, text="Result:", foreground=TK_SUBTEXT).grid(row=8, column=0, sticky="w")
        ttk.Label(recog_frame, textvariable=self._var_last_result).grid(row=8, column=1, sticky="w")

        ttk.Label(recog_frame, text="Record ID:", foreground=TK_SUBTEXT).grid(row=9, column=0, sticky="w")
        ttk.Label(recog_frame, textvariable=self._var_last_record_id,
                  font=("Consolas", 9)).grid(row=9, column=1, sticky="w")

        # Danh sách lịch sử
        hist_frame = ttk.LabelFrame(side, text=" 📋  Lịch sử sự kiện ", padding=4)
        hist_frame.grid(row=2, column=0, sticky="nsew")
        hist_frame.rowconfigure(0, weight=1)
        hist_frame.columnconfigure(0, weight=1)

        self._listbox = tk.Listbox(hist_frame, font=("Consolas", 9), bg="#1a1a2e",
                                    fg=TK_TEXT, selectbackground=TK_ACCENT,
                                    relief="flat", borderwidth=0)
        self._listbox.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(hist_frame, orient="vertical", command=self._listbox.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._listbox.configure(yscrollcommand=sb.set)

    # ------------------------------------------------------------------
    # Điều khiển camera
    # ------------------------------------------------------------------

    def _start(self) -> None:
        if self._running:
            return
        if not self._engine.ready:
            messagebox.showwarning("Chưa sẵn sàng", "AI engine chưa tải xong. Vui lòng chờ.")
            return
        self._running = True
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._attendance = AttendanceService(
            cooldown_seconds=self.COOLDOWN_SECONDS,
            min_toggle_seconds=self.MIN_TOGGLE_SECONDS,
        )
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

    def _stop(self) -> None:
        self._running = False
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._status_var.set("⏸ Camera đã dừng")

    def _reset(self) -> None:
        self._attendance.reset_state()
        self._status_var.set("↺ Đã reset cooldown/tracking")

    def _export(self) -> None:
        dirs = self._attendance.export_by_date(ATTENDANCE_DIR)
        paths_txt = "\n".join(str(d) for d in dirs)
        messagebox.showinfo("Xuất log", f"Đã lưu log theo ngày (Năm/Tháng/Ngày):\n{paths_txt}")

    # ------------------------------------------------------------------
    # Vòng lặp camera (chạy trong thread riêng)
    # ------------------------------------------------------------------

    def _camera_loop(self) -> None:
        try:
            with CameraCapture(camera_index=0) as cam:
                cam_info = cam.get_info()
                cam_w, cam_h = cam_info["width"], cam_info["height"]
                self._status_var.set(f"▶ Đang chạy — Camera {cam_w}×{cam_h}")

                while self._running:
                    frame = cam.read_frame()
                    display = frame.copy()

                    try:
                        detections = self._engine.detector.detect(frame)
                    except Exception:
                        detections = []

                    best = FaceDetector.largest_face(detections)
                    match_result = None

                    if best is not None:
                        try:
                            if best.landmarks is not None:
                                aligned_img = align_face(frame, best.landmarks).image
                            else:
                                aligned_img = crop_face_simple(frame, best.bbox)

                            qr = self._engine.checker.check(
                                aligned_img,
                                confidence=best.confidence,
                                face_bbox=best.bbox,
                                landmarks=best.landmarks,
                            )

                            if qr.passed:
                                emb = self._engine.recognizer.get_embedding(aligned_img)
                                match_result = self._engine.matcher.match(emb)
                        except Exception:
                            pass

                        # Vẽ bounding box
                        x1, y1, x2, y2 = best.as_int_bbox()
                        x1c = max(0, x1); y1c = max(0, y1)
                        x2c = min(cam_w - 1, x2); y2c = min(cam_h - 1, y2)

                        if match_result is not None:
                            box_color = _status_color(match_result.status)
                            if x2c > x1c and y2c > y1c:
                                cv2.rectangle(display, (x1c, y1c), (x2c, y2c), box_color, 2)

                            # Nhãn "Mã - Tên"
                            if match_result.employee_id:
                                if match_result.employee_name:
                                    label = f"{match_result.employee_id} - {match_result.employee_name}"
                                else:
                                    label = match_result.employee_id
                            else:
                                label = match_result.status.value.upper()

                            sim_txt = f"  ({match_result.similarity:.2f})"
                            _cv_put(display, label + sim_txt,
                                    (x1c + 2, max(y1c - 8, 14)), box_color, 0.55, 2)

                            # Chấm công
                            decision = self._attendance.process(
                                match_result, camera_id=self.CAMERA_ID
                            )
                            if decision.created:
                                ev = decision.event
                                icon = "CHECK-IN" if ev.event_type == EventType.CHECK_IN else "CHECK-OUT"
                                punct = ""
                                if ev.event_type == EventType.CHECK_IN:
                                    punct = f" | {ev.punctuality_status.display_vi}"
                                    if ev.late_minutes:
                                        punct += f" ({ev.late_minutes}p)"
                                line = f"[{icon}] {ev.employee_id} | {_now_local()}{punct}"
                                self._add_event(line, ev)
                        else:
                            box_color = CV_RED
                            if x2c > x1c and y2c > y1c:
                                cv2.rectangle(display, (x1c, y1c), (x2c, y2c), box_color, 1)

                    # HUD
                    _cv_put(display,
                            f"Gallery: {self._engine.gallery.size} NV  |  SuKien: {self._attendance.total_events}",
                            (10, 22), CV_CYAN, 0.5)
                    _cv_put(display, f"Faces: {len(detections)}", (10, 42), CV_WHITE, 0.42)

                    # Cập nhật canvas
                    self._update_canvas(display)

        except RuntimeError as exc:
            self._status_var.set(f"❌ Lỗi camera: {exc}")
        finally:
            self._running = False
            self.after(0, lambda: self._btn_start.config(state="normal"))
            self.after(0, lambda: self._btn_stop.config(state="disabled"))

    def _update_canvas(self, frame_bgr: np.ndarray) -> None:
        """Convert BGR frame → Tkinter ImageTk và cập nhật canvas."""
        if not PIL_AVAILABLE:
            return
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Scale để vừa canvas
            canvas_w = self._canvas.winfo_width() or 640
            canvas_h = self._canvas.winfo_height() or 480
            h, w = rgb.shape[:2]
            scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            rgb_resized = cv2.resize(rgb, (new_w, new_h))
            img = Image.fromarray(rgb_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.after(0, self._set_canvas_image, imgtk, canvas_w, canvas_h, new_w, new_h)
        except Exception:
            pass

    def _set_canvas_image(self, imgtk, cw, ch, iw, ih) -> None:
        self._canvas.imgtk = imgtk  # giữ reference
        x = (cw - iw) // 2
        y = (ch - ih) // 2
        self._canvas.create_image(x, y, anchor="nw", image=imgtk)

    def _add_event(self, line: str, ev) -> None:
        """Cập nhật lịch sử và panel nhận diện gần nhất (thread-safe via after())."""
        name = self._engine.gallery.get_name(ev.employee_id)

        def _update():
            # Thêm vào listbox
            self._listbox.insert(0, line)
            if self._listbox.size() > 50:
                self._listbox.delete(50, tk.END)

            # Thẻ nhận diện gần nhất — đầy đủ theo cấu trúc nghiệp vụ
            self._var_last_id.set(ev.employee_id)
            self._var_last_name.set(name or "—")
            self._var_last_time.set(_now_local())
            event_label = "✅ Check-in" if ev.event_type == EventType.CHECK_IN else "🚪 Check-out"
            self._var_last_type.set(event_label)

            # Trạng thái đúng/trễ giờ (chỉ cho CHECK_IN)
            if ev.event_type == EventType.CHECK_IN:
                punct = ev.punctuality_status.display_vi
                late_txt = f" ({ev.late_minutes} phút)" if ev.late_minutes else ""
                self._var_last_punctuality.set(f"{punct}{late_txt}")
            else:
                self._var_last_punctuality.set("—")

            self._var_last_device.set(ev.camera_id)
            self._var_last_method.set(ev.verify_method)
            self._var_last_vstatus.set(ev.verify_status)
            self._var_last_result.set(ev.result)
            self._var_last_record_id.set(str(ev.event_id))

            # Thống kê
            self._var_total.set(str(self._attendance.total_events))
            evts = self._attendance.events
            self._var_in.set(str(sum(1 for e in evts if e.event_type == EventType.CHECK_IN)))
            self._var_out.set(str(sum(1 for e in evts if e.event_type == EventType.CHECK_OUT)))
            self._var_gallery.set(str(self._engine.gallery.size))

        self.after(0, _update)


# ---------------------------------------------------------------------------
# Tab 2: Đăng Ký Nhân Viên
# ---------------------------------------------------------------------------

class EnrollmentTab(ttk.Frame):
    """Tab đăng ký: form nhập thông tin + camera preview + finalize."""

    def __init__(self, parent: ttk.Notebook, engine: AIEngine) -> None:
        super().__init__(parent)
        self._engine   = engine
        self._service: Optional[EnrollmentService] = None
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._current_emp_id: str = ""
        self._pending_count  = 0
        self._lock = threading.Lock()

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # --- Camera panel ---
        cam_frame = ttk.LabelFrame(self, text=" 📷  Camera đăng ký ", padding=4)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        cam_frame.rowconfigure(0, weight=1)
        cam_frame.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(cam_frame, bg="#000", width=640, height=480)
        self._canvas.grid(row=0, column=0, sticky="nsew")

        btn_row = ttk.Frame(cam_frame)
        btn_row.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self._btn_capture = ttk.Button(btn_row, text="📸  Chụp mẫu [SPACE]",
                                        command=self._capture, state="disabled")
        self._btn_capture.pack(side="left", padx=4)

        self._btn_finalize = ttk.Button(btn_row, text="✅  Hoàn tất đăng ký",
                                         command=self._finalize, state="disabled")
        self._btn_finalize.pack(side="left", padx=4)

        ttk.Button(btn_row, text="↺  Reset mẫu", command=self._reset_pending).pack(side="left", padx=4)

        self._cam_status = tk.StringVar(value="⏸ Camera chưa bật")
        ttk.Label(cam_frame, textvariable=self._cam_status, foreground=TK_SUBTEXT).grid(
            row=2, column=0, sticky="w", pady=(2, 0)
        )

        self._quality_var = tk.StringVar(value="")
        ttk.Label(cam_frame, textvariable=self._quality_var).grid(
            row=3, column=0, sticky="w"
        )

        # --- Side panel ---
        side = ttk.Frame(self)
        side.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(3, weight=1)

        # Form nhập thông tin
        form = ttk.LabelFrame(side, text=" 👤  Thông tin nhân viên ", padding=10)
        form.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Mã NV (tự động):").grid(row=0, column=0, sticky="w", pady=2)
        self._var_emp_id = tk.StringVar(value="(chưa tạo)")
        ttk.Label(form, textvariable=self._var_emp_id,
                  font=("Consolas", 10, "bold"), foreground=TK_ACCENT).grid(
            row=0, column=1, sticky="w", pady=2
        )

        ttk.Label(form, text="Họ tên *:").grid(row=1, column=0, sticky="w", pady=2)
        self._entry_name = ttk.Entry(form, width=22)
        self._entry_name.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(form, text="Phòng ban:").grid(row=2, column=0, sticky="w", pady=2)
        self._entry_dept = ttk.Entry(form, width=22)
        self._entry_dept.grid(row=2, column=1, sticky="ew", pady=2)

        self._btn_new = ttk.Button(form, text="➕  Tạo hồ sơ & Mở camera",
                                    command=self._create_profile)
        self._btn_new.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        # Progress
        prog_frame = ttk.LabelFrame(side, text=" 📊  Tiến trình thu mẫu ", padding=8)
        prog_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        prog_frame.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress = ttk.Progressbar(prog_frame, variable=self._progress_var,
                                          maximum=MIN_SAMPLES_REQUIRED, length=200)
        self._progress.grid(row=0, column=0, sticky="ew")

        self._var_progress_txt = tk.StringVar(value=f"0 / {MIN_SAMPLES_REQUIRED} mẫu")
        ttk.Label(prog_frame, textvariable=self._var_progress_txt).grid(
            row=1, column=0, sticky="w"
        )

        # Trạng thái mẫu gần nhất
        self._var_sample_msg = tk.StringVar(value="")
        lbl_sample = ttk.Label(prog_frame, textvariable=self._var_sample_msg,
                                wraplength=200)
        lbl_sample.grid(row=2, column=0, sticky="w")

        # Danh sách gallery
        gal_frame = ttk.LabelFrame(side, text=" 🗂️  Gallery nhân viên ", padding=4)
        gal_frame.grid(row=3, column=0, sticky="nsew")
        gal_frame.rowconfigure(0, weight=1)
        gal_frame.columnconfigure(0, weight=1)

        self._gal_listbox = tk.Listbox(gal_frame, font=("Consolas", 9),
                                        bg="#1a1a2e", fg=TK_TEXT,
                                        selectbackground=TK_ACCENT,
                                        relief="flat", borderwidth=0)
        self._gal_listbox.grid(row=0, column=0, sticky="nsew")

        sb2 = ttk.Scrollbar(gal_frame, orient="vertical", command=self._gal_listbox.yview)
        sb2.grid(row=0, column=1, sticky="ns")
        self._gal_listbox.configure(yscrollcommand=sb2.set)

        ttk.Button(gal_frame, text="🔄  Làm mới gallery", command=self._refresh_gallery).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )

        self._refresh_gallery()

        # Bind phím Space
        self.bind_all("<space>", lambda e: self._capture())

    # ------------------------------------------------------------------
    # Hành động người dùng
    # ------------------------------------------------------------------

    def _create_profile(self) -> None:
        name = self._entry_name.get().strip()
        dept = self._entry_dept.get().strip()

        if not name:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập họ tên nhân viên.")
            return

        if not self._engine.ready:
            messagebox.showwarning("Chưa sẵn sàng", "AI engine chưa tải xong.")
            return

        # Tạo EnrollmentService mới cho mỗi phiên
        gallery = self._engine.gallery
        from app.vision.face_matcher import FaceGallery as _FG
        self._service = EnrollmentService(
            detector=self._engine.detector,
            recognizer=self._engine.recognizer,
            gallery=gallery,
            quality_checker=self._engine.checker,
            min_samples=MIN_SAMPLES_REQUIRED,
            max_samples=10,
            min_consistency=0.10,
            save_images_dir=FACES_DIR,
        )

        from app.enrollment.enrollment_service import EnrollmentService as _ES
        emp_id = _ES.generate_employee_id(COMPANY_CODE)

        try:
            self._service.create_profile(emp_id, name, department=dept)
        except ValueError as exc:
            messagebox.showerror("Lỗi", str(exc))
            return

        self._current_emp_id = emp_id
        self._var_emp_id.set(emp_id)
        self._pending_count = 0
        self._progress_var.set(0)
        self._var_progress_txt.set(f"0 / {MIN_SAMPLES_REQUIRED} mẫu")
        self._var_sample_msg.set("")

        # Bật camera
        if not self._running:
            self._running = True
            self._btn_capture.config(state="normal")
            self._thread = threading.Thread(target=self._camera_loop, daemon=True)
            self._thread.start()

        self._cam_status.set(f"▶ Đang thu mẫu cho: {emp_id} — {name}")

    def _capture(self) -> None:
        if not self._service or not self._current_emp_id:
            return
        # Lấy frame gần nhất (đọc từ biến shared)
        with self._lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
        if frame is None:
            return
        result = self._service.add_sample(self._current_emp_id, frame)
        self._pending_count = self._service.pending_count(self._current_emp_id)
        self._progress_var.set(min(self._pending_count, MIN_SAMPLES_REQUIRED))
        self._var_progress_txt.set(f"{self._pending_count} / {MIN_SAMPLES_REQUIRED} mẫu")
        self._var_sample_msg.set(result.message)

        if self._pending_count >= MIN_SAMPLES_REQUIRED:
            self._btn_finalize.config(state="normal")

    def _finalize(self) -> None:
        if not self._service or not self._current_emp_id:
            return
        final = self._service.finalize(self._current_emp_id)
        if final.success:
            # Lưu tên vào gallery
            profile = self._service.get_profile(self._current_emp_id)
            if profile:
                self._engine.gallery.set_name(self._current_emp_id, profile.name)
            self._engine.gallery.save(GALLERY_FILE)
            self._engine.reload_gallery()
            self._refresh_gallery()
            messagebox.showinfo("Thành công",
                                f"✅ {final.message}\n\nMã NV: {self._current_emp_id}")
            self._stop_camera()
            self._btn_finalize.config(state="disabled")
            self._var_emp_id.set("(chưa tạo)")
            self._entry_name.delete(0, tk.END)
            self._entry_dept.delete(0, tk.END)
        else:
            messagebox.showwarning("Chưa đủ mẫu", final.message)

    def _reset_pending(self) -> None:
        if self._service and self._current_emp_id:
            self._service.reset_pending(self._current_emp_id)
            self._pending_count = 0
            self._progress_var.set(0)
            self._var_progress_txt.set(f"0 / {MIN_SAMPLES_REQUIRED} mẫu")
            self._var_sample_msg.set("Đã reset tất cả mẫu chưa lưu.")
            self._btn_finalize.config(state="disabled")

    def _stop_camera(self) -> None:
        self._running = False
        self._btn_capture.config(state="disabled")

    def _refresh_gallery(self) -> None:
        self._engine.reload_gallery()
        self._gal_listbox.delete(0, tk.END)
        for emp_id in self._engine.gallery.employee_ids:
            name = self._engine.gallery.get_name(emp_id) or "?"
            n_samples = len(self._engine.gallery.get_embeddings(emp_id))
            self._gal_listbox.insert(tk.END, f"{emp_id}  |  {name}  ({n_samples} mẫu)")

    # ------------------------------------------------------------------
    # Vòng lặp camera
    # ------------------------------------------------------------------

    def _camera_loop(self) -> None:
        try:
            with CameraCapture(camera_index=0) as cam:
                cam_info = cam.get_info()
                cam_w, cam_h = cam_info["width"], cam_info["height"]
                self._cam_status.set(f"▶ Camera {cam_w}×{cam_h} — sẵn sàng chụp mẫu")

                while self._running:
                    frame = cam.read_frame()
                    with self._lock:
                        self._latest_frame = frame.copy()

                    display = frame.copy()

                    try:
                        detections = self._engine.detector.detect(frame)
                    except Exception:
                        detections = []

                    best = FaceDetector.largest_face(detections)
                    quality_ok = False

                    if best is not None:
                        x1, y1, x2, y2 = best.as_int_bbox()
                        x1c = max(0, x1); y1c = max(0, y1)
                        x2c = min(cam_w - 1, x2); y2c = min(cam_h - 1, y2)

                        try:
                            if best.landmarks is not None:
                                aligned_img = align_face(frame, best.landmarks).image
                            else:
                                aligned_img = crop_face_simple(frame, best.bbox)
                            qr = self._engine.checker.check(
                                aligned_img,
                                confidence=best.confidence,
                                face_bbox=best.bbox,
                                landmarks=best.landmarks,
                            )
                            quality_ok = qr.passed
                            guide_msg = qr.message if not quality_ok else "✓ Chất lượng OK — nhấn SPACE để chụp"
                        except Exception:
                            guide_msg = "Lỗi quality check"

                        box_color = CV_GREEN if quality_ok else CV_RED
                        if x2c > x1c and y2c > y1c:
                            cv2.rectangle(display, (x1c, y1c), (x2c, y2c), box_color, 2)
                        _cv_put(display, f"{best.confidence:.2f}",
                                (x1c + 2, max(y1c - 6, 14)), box_color, 0.5)

                        self.after(0, lambda m=guide_msg: self._quality_var.set(m))

                    # Progress overlay
                    _cv_put(display,
                            f"Mau: {self._pending_count}/{MIN_SAMPLES_REQUIRED}",
                            (10, 25), CV_CYAN, 0.6, 2)
                    _cv_put(display, "[SPACE]=chup  [Finalize]=luu",
                            (10, cam_h - 10), CV_GRAY, 0.38)

                    self._update_canvas(display)

        except RuntimeError as exc:
            self._cam_status.set(f"❌ Lỗi camera: {exc}")
        finally:
            self._running = False

    def _update_canvas(self, frame_bgr: np.ndarray) -> None:
        if not PIL_AVAILABLE:
            return
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            canvas_w = self._canvas.winfo_width() or 640
            canvas_h = self._canvas.winfo_height() or 480
            h, w = rgb.shape[:2]
            scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            rgb_resized = cv2.resize(rgb, (new_w, new_h))
            img = Image.fromarray(rgb_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.after(0, self._set_canvas_image, imgtk, canvas_w, canvas_h, new_w, new_h)
        except Exception:
            pass

    def _set_canvas_image(self, imgtk, cw, ch, iw, ih) -> None:
        self._canvas.imgtk = imgtk
        x = (cw - iw) // 2
        y = (ch - ih) // 2
        self._canvas.create_image(x, y, anchor="nw", image=imgtk)


# ---------------------------------------------------------------------------
# Tab 3: Gallery Manager
# ---------------------------------------------------------------------------

class GalleryTab(ttk.Frame):
    """Tab xem và quản lý gallery nhân viên đã đăng ký."""

    def __init__(self, parent: ttk.Notebook, engine: AIEngine) -> None:
        super().__init__(parent)
        self._engine = engine
        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=6)

        ttk.Button(toolbar, text="🔄  Làm mới", command=self._refresh).pack(side="left", padx=4)
        ttk.Button(toolbar, text="🗑️  Xóa nhân viên được chọn",
                   command=self._delete_selected).pack(side="left", padx=4)

        self._var_count = tk.StringVar()
        ttk.Label(toolbar, textvariable=self._var_count, foreground=TK_SUBTEXT).pack(
            side="right", padx=8
        )

        # Bảng danh sách
        tbl_frame = ttk.Frame(self)
        tbl_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        tbl_frame.rowconfigure(0, weight=1)
        tbl_frame.columnconfigure(0, weight=1)

        cols = ("Mã NV", "Họ tên", "Số mẫu")
        self._tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", selectmode="browse")
        for col in cols:
            self._tree.heading(col, text=col)
        self._tree.column("Mã NV",  width=160, anchor="w")
        self._tree.column("Họ tên", width=220, anchor="w")
        self._tree.column("Số mẫu", width=80,  anchor="center")
        self._tree.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(tbl_frame, orient="vertical", command=self._tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=sb.set)

        self._refresh()

    def _refresh(self) -> None:
        self._engine.reload_gallery()
        for item in self._tree.get_children():
            self._tree.delete(item)
        for emp_id in self._engine.gallery.employee_ids:
            name = self._engine.gallery.get_name(emp_id) or "—"
            n = len(self._engine.gallery.get_embeddings(emp_id))
            self._tree.insert("", tk.END, values=(emp_id, name, n))
        self._var_count.set(f"Tổng: {self._engine.gallery.size} nhân viên")

    def _delete_selected(self) -> None:
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Thông báo", "Chưa chọn nhân viên nào.")
            return
        vals = self._tree.item(sel[0], "values")
        emp_id = vals[0]
        name   = vals[1]
        if not messagebox.askyesno("Xác nhận xóa",
                                    f"Xóa nhân viên:\n  {emp_id} — {name}\n\nHành động này không thể hoàn tác!"):
            return
        self._engine.gallery.remove(emp_id)
        self._engine.gallery.save(GALLERY_FILE)
        self._engine.reload_gallery()
        self._refresh()
        messagebox.showinfo("Đã xóa", f"Đã xóa {emp_id} — {name} khỏi gallery.")


# ---------------------------------------------------------------------------
# Cửa sổ chính
# ---------------------------------------------------------------------------

class MainWindow(tk.Tk):
    """Cửa sổ chính của ứng dụng FaceAttend."""

    def __init__(self) -> None:
        super().__init__()
        self.title("FaceAttend — Hệ thống chấm công khuôn mặt")
        self.geometry("1200x750")
        self.minsize(900, 600)

        self._engine = AIEngine()
        self._setup_theme()
        self._build_ui()
        self._load_engine_async()

    def _setup_theme(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        self.configure(bg=TK_BG)
        style.configure(".", background=TK_BG, foreground=TK_TEXT,
                         fieldbackground=TK_PANEL, bordercolor="#444",
                         troughcolor=TK_PANEL)
        style.configure("TNotebook",       background=TK_BG, tabmargins=[2, 4, 2, 0])
        style.configure("TNotebook.Tab",   background=TK_PANEL, foreground=TK_TEXT,
                         padding=[14, 6], font=("Arial", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", TK_ACCENT)],
                  foreground=[("selected", "#fff")])
        style.configure("TLabelframe",     background=TK_BG, foreground=TK_SUBTEXT)
        style.configure("TLabelframe.Label", background=TK_BG, foreground=TK_SUBTEXT,
                         font=("Arial", 9, "bold"))
        style.configure("TButton",         background=TK_PANEL, foreground=TK_TEXT,
                         padding=[8, 4], relief="flat")
        style.map("TButton",
                  background=[("active", TK_ACCENT), ("disabled", "#333")],
                  foreground=[("disabled", "#666")])
        style.configure("TEntry",          background=TK_PANEL, foreground=TK_TEXT,
                         insertcolor=TK_TEXT)
        style.configure("Treeview",        background=TK_PANEL, foreground=TK_TEXT,
                         fieldbackground=TK_PANEL, rowheight=26)
        style.configure("Treeview.Heading", background="#333", foreground=TK_TEXT,
                         font=("Arial", 9, "bold"))
        style.map("Treeview", background=[("selected", TK_ACCENT)])
        style.configure("TProgressbar",    troughcolor=TK_PANEL, background=TK_GREEN)

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self, bg="#120d20", height=52)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header, text="🎯  FaceAttend", font=("Arial", 16, "bold"),
                 bg="#120d20", fg=TK_ACCENT).pack(side="left", padx=16, pady=8)
        tk.Label(header, text="Hệ thống chấm công nhận diện khuôn mặt",
                 font=("Arial", 9), bg="#120d20", fg=TK_SUBTEXT).pack(side="left", pady=8)

        # Status bar
        self._status_bar = tk.Label(
            self, text="⏳ Đang khởi động...", anchor="w",
            bg="#0a0a1a", fg=TK_SUBTEXT, font=("Arial", 8), padx=8
        )
        self._status_bar.pack(fill="x", side="bottom")

        # Notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self._tab_attend = AttendanceTab(nb, self._engine)
        nb.add(self._tab_attend, text="  📹  Điểm Danh  ")

        self._tab_enroll = EnrollmentTab(nb, self._engine)
        nb.add(self._tab_enroll, text="  👤  Đăng Ký  ")

        self._tab_gallery = GalleryTab(nb, self._engine)
        nb.add(self._tab_gallery, text="  🗂️  Gallery  ")

    def _load_engine_async(self) -> None:
        def _do_load():
            self._engine.load(progress_cb=self._on_engine_progress)

        t = threading.Thread(target=_do_load, daemon=True)
        t.start()

    def _on_engine_progress(self, msg: str) -> None:
        self.after(0, lambda: self._status_bar.config(text=f"  {msg}"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not PIL_AVAILABLE:
        import tkinter.messagebox as mb
        import tkinter as _tk
        r = _tk.Tk()
        r.withdraw()
        mb.showwarning(
            "Thiếu thư viện",
            "Vui lòng cài Pillow để hiển thị video:\n\n"
            "  .venv\\Scripts\\pip install pillow\n\nSau đó chạy lại."
        )
        r.destroy()
        return

    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
