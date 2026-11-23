"""
app/database/repositories/employee_repo.py
Data-access layer cho bảng employees — Giai đoạn 9.

Mọi thao tác DB liên quan đến nhân viên đi qua class này.
Service layer (enrollment_service, attendance_service) gọi repo thay vì
viết SQL/ORM trực tiếp.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Employee

logger = logging.getLogger(__name__)


class EmployeeRepository:
    """CRUD operations cho bảng employees."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        employee_code: str,
        full_name: str,
        email: Optional[str] = None,
        department: Optional[str] = None,
        position: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Employee:
        """
        Tạo nhân viên mới.
        Raise ValueError nếu employee_code hoặc email đã tồn tại.
        """
        if self.get_by_code(employee_code):
            raise ValueError(f"employee_code đã tồn tại: {employee_code!r}")
        if email and self.get_by_email(email):
            raise ValueError(f"email đã tồn tại: {email!r}")

        emp = Employee(
            employee_code=employee_code,
            full_name=full_name,
            email=email,
            department=department,
            position=position,
            notes=notes,
            status="active",
        )
        self._session.add(emp)
        self._session.flush()  # lấy id mà chưa commit
        logger.info("Tạo nhân viên: %s (%s)", employee_code, full_name)
        return emp

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, employee_id: int) -> Optional[Employee]:
        """Lấy nhân viên theo PK."""
        return self._session.get(Employee, employee_id)

    def get_by_code(self, employee_code: str) -> Optional[Employee]:
        """Lấy nhân viên theo mã nghiệp vụ."""
        stmt = select(Employee).where(Employee.employee_code == employee_code)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_by_email(self, email: str) -> Optional[Employee]:
        """Lấy nhân viên theo email."""
        stmt = select(Employee).where(Employee.email == email)
        return self._session.execute(stmt).scalar_one_or_none()

    def list_active(self) -> list[Employee]:
        """Lấy tất cả nhân viên đang active."""
        stmt = select(Employee).where(Employee.status == "active").order_by(Employee.full_name)
        return list(self._session.execute(stmt).scalars().all())

    def list_all(self, include_deleted: bool = False) -> list[Employee]:
        """Lấy toàn bộ nhân viên; mặc định bỏ qua deleted."""
        stmt = select(Employee).order_by(Employee.full_name)
        if not include_deleted:
            stmt = stmt.where(Employee.status != "deleted")
        return list(self._session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        employee_id: int,
        full_name: Optional[str] = None,
        email: Optional[str] = None,
        department: Optional[str] = None,
        position: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Employee:
        """Cập nhật thông tin nhân viên. Raise ValueError nếu không tìm thấy."""
        emp = self.get_by_id(employee_id)
        if emp is None:
            raise ValueError(f"Không tìm thấy nhân viên id={employee_id}")

        if full_name is not None:
            emp.full_name = full_name
        if email is not None:
            emp.email = email
        if department is not None:
            emp.department = department
        if position is not None:
            emp.position = position
        if notes is not None:
            emp.notes = notes

        self._session.flush()
        logger.info("Cập nhật nhân viên id=%d", employee_id)
        return emp

    def set_status(self, employee_id: int, status: str) -> Employee:
        """
        Đổi trạng thái nhân viên.
        status: 'active' | 'inactive' | 'deleted'
        """
        valid = {"active", "inactive", "deleted"}
        if status not in valid:
            raise ValueError(f"status không hợp lệ: {status!r}. Phải là {valid}")

        emp = self.get_by_id(employee_id)
        if emp is None:
            raise ValueError(f"Không tìm thấy nhân viên id={employee_id}")

        emp.status = status
        self._session.flush()
        logger.info("Đổi trạng thái nhân viên id=%d → %s", employee_id, status)
        return emp

    # ------------------------------------------------------------------
    # Delete (hard delete — chỉ dùng khi thực sự cần xóa toàn bộ dữ liệu)
    # ------------------------------------------------------------------

    def hard_delete(self, employee_id: int) -> bool:
        """
        Xóa hẳn bản ghi nhân viên khỏi DB.
        WARNING: Dùng soft-delete (set_status 'deleted') thay thế trong hầu hết trường hợp.
        Trả True nếu xóa thành công, False nếu không tìm thấy.
        """
        emp = self.get_by_id(employee_id)
        if emp is None:
            return False
        self._session.delete(emp)
        self._session.flush()
        logger.warning("HARD DELETE nhân viên id=%d code=%s", employee_id, emp.employee_code)
        return True
