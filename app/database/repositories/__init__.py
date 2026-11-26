"""
app/database/repositories/__init__.py
"""
from app.database.repositories.employee_repo import EmployeeRepository
from app.database.repositories.face_template_repo import FaceTemplateRepository
from app.database.repositories.attendance_repo import AttendanceRepository
from app.database.repositories.attendance_record_repo import AttendanceRecordRepository
from app.database.repositories.face_recognition_log_repo import FaceRecognitionLogRepository
from app.database.repositories.shift_repo import ShiftRepository

__all__ = [
    "EmployeeRepository",
    "FaceTemplateRepository",
    "AttendanceRepository",
    "AttendanceRecordRepository",
    "FaceRecognitionLogRepository",
    "ShiftRepository",
]
