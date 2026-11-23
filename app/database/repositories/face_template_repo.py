"""
app/database/repositories/face_template_repo.py
Data-access layer cho bảng face_templates — Giai đoạn 9.

Embedding là dữ liệu sinh trắc học nhạy cảm.
Hạn chế log; chỉ truy cập khi thật sự cần cho nghiệp vụ.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import FaceTemplate, embedding_to_bytes, bytes_to_embedding, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class FaceTemplateRepository:
    """CRUD operations cho bảng face_templates."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def add(
        self,
        employee_id: int,
        embedding: np.ndarray,
        model_name: str,
        model_version: str = "1.0",
        quality_score: Optional[float] = None,
        expires_at: Optional[datetime] = None,
    ) -> FaceTemplate:
        """
        Thêm một mẫu embedding cho nhân viên.
        embedding: numpy array shape (512,) dtype float32, đã L2-normalize.
        """
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(f"Embedding phải có shape ({EMBEDDING_DIM},), nhận {embedding.shape}")
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding chứa NaN hoặc Inf")

        tpl = FaceTemplate(
            employee_id=employee_id,
            embedding=embedding_to_bytes(embedding),
            model_name=model_name,
            model_version=model_version,
            quality_score=quality_score,
            is_active=True,
            expires_at=expires_at,
        )
        self._session.add(tpl)
        self._session.flush()
        logger.info(
            "Thêm FaceTemplate id=%d employee_id=%d model=%s@%s",
            tpl.id, employee_id, model_name, model_version,
        )
        return tpl

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, template_id: int) -> Optional[FaceTemplate]:
        """Lấy một template theo PK."""
        return self._session.get(FaceTemplate, template_id)

    def list_active_by_employee(
        self,
        employee_id: int,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> list[FaceTemplate]:
        """
        Lấy tất cả template đang active của một nhân viên.
        Nếu model_name/model_version được chỉ định, lọc theo model đó.
        """
        stmt = select(FaceTemplate).where(
            FaceTemplate.employee_id == employee_id,
            FaceTemplate.is_active.is_(True),
        )
        if model_name:
            stmt = stmt.where(FaceTemplate.model_name == model_name)
        if model_version:
            stmt = stmt.where(FaceTemplate.model_version == model_version)
        return list(self._session.execute(stmt).scalars().all())

    def load_gallery(
        self,
        model_name: str,
        model_version: str,
        only_active_employees: bool = True,
    ) -> dict[int, list[np.ndarray]]:
        """
        Nạp toàn bộ gallery để dùng với FaceMatcher.

        Trả về: {employee_id: [embedding1, embedding2, ...]}
        Chỉ bao gồm template is_active=True.
        Nếu only_active_employees=True, chỉ lấy nhân viên có status='active'.

        NOTE: Method này đọc embedding từ DB lên RAM.
        Không gọi quá thường xuyên; nên cache kết quả ở tầng service.
        """
        from app.database.models import Employee  # import cục bộ tránh circular

        stmt = (
            select(FaceTemplate)
            .where(
                FaceTemplate.is_active.is_(True),
                FaceTemplate.model_name == model_name,
                FaceTemplate.model_version == model_version,
            )
        )
        if only_active_employees:
            stmt = stmt.join(Employee).where(Employee.status == "active")

        templates = self._session.execute(stmt).scalars().all()

        gallery: dict[int, list[np.ndarray]] = {}
        for tpl in templates:
            emb = bytes_to_embedding(tpl.embedding)
            gallery.setdefault(tpl.employee_id, []).append(emb)

        logger.debug(
            "Nạp gallery: %d nhân viên, %d template tổng (model=%s@%s)",
            len(gallery), len(templates), model_name, model_version,
        )
        return gallery

    def count_by_employee(self, employee_id: int, active_only: bool = True) -> int:
        """Đếm số template của một nhân viên."""
        from sqlalchemy import func
        stmt = select(func.count()).select_from(FaceTemplate).where(
            FaceTemplate.employee_id == employee_id
        )
        if active_only:
            stmt = stmt.where(FaceTemplate.is_active.is_(True))
        return self._session.execute(stmt).scalar_one()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def deactivate(self, template_id: int) -> bool:
        """Vô hiệu hóa một template (is_active=False). Trả True nếu thành công."""
        tpl = self.get_by_id(template_id)
        if tpl is None:
            return False
        tpl.is_active = False
        self._session.flush()
        logger.info("Vô hiệu hóa FaceTemplate id=%d employee_id=%d", template_id, tpl.employee_id)
        return True

    def deactivate_all_by_employee(self, employee_id: int) -> int:
        """
        Vô hiệu hóa tất cả template của một nhân viên.
        Trả số template bị ảnh hưởng.
        """
        templates = self.list_active_by_employee(employee_id)
        for tpl in templates:
            tpl.is_active = False
        self._session.flush()
        logger.info(
            "Vô hiệu hóa %d template của employee_id=%d", len(templates), employee_id
        )
        return len(templates)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_by_employee(self, employee_id: int) -> int:
        """
        Xóa hẳn tất cả template của nhân viên khỏi DB.
        Dùng khi nhân viên yêu cầu xóa dữ liệu sinh trắc học.
        Trả số bản ghi đã xóa.
        """
        stmt = select(FaceTemplate).where(FaceTemplate.employee_id == employee_id)
        templates = list(self._session.execute(stmt).scalars().all())
        for tpl in templates:
            self._session.delete(tpl)
        self._session.flush()
        logger.warning(
            "Xóa %d FaceTemplate của employee_id=%d", len(templates), employee_id
        )
        return len(templates)
