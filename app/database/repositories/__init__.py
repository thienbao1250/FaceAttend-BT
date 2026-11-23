"""
app/database/repositories/__init__.py
"""
from app.database.repositories.employee_repo import EmployeeRepository
from app.database.repositories.face_template_repo import FaceTemplateRepository
from app.database.repositories.attendance_repo import AttendanceRepository

__all__ = [
    "EmployeeRepository",
    "FaceTemplateRepository",
    "AttendanceRepository",
]
