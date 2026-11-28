"""
app/database/connection.py
Quản lý kết nối database — Giai đoạn 9.

Hỗ trợ:
  - PostgreSQL (production): đọc từ DATABASE_URL trong .env
  - SQLite (fallback local/test): dùng khi không có DATABASE_URL

Dùng SQLAlchemy 2.0 (style mới: Session với context manager).

Cách dùng:
    from app.database.connection import get_session

    with get_session() as session:
        result = session.execute(select(Employee)).scalars().all()

Môi trường:
    DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/faceattend
    # Nếu không set → SQLite tại data/faceattend.db (chỉ dùng dev/test)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Tự động nạp biến môi trường từ file .env nếu có python-dotenv
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parents[2] / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
        logging.getLogger(__name__).debug("Đã nạp .env từ %s", _env_file)
except ImportError:
    pass  # python-dotenv chưa cài — dùng biến môi trường hệ thống bình thường

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class cho tất cả ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Base khai báo ORM chung cho toàn bộ models."""
    pass


# ---------------------------------------------------------------------------
# Xác định DATABASE_URL
# ---------------------------------------------------------------------------

def _get_database_url() -> str:
    """
    Đọc DATABASE_URL từ biến môi trường / file .env.
    Bắt buộc phải cấu hình PostgreSQL (không dùng SQLite fallback).
    """
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL chưa được cấu hình. "
            "Vui lòng thiết lập DATABASE_URL trong file .env (PostgreSQL)."
        )
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    logger.info("Database: PostgreSQL (từ DATABASE_URL)")
    return url


DATABASE_URL: str = _get_database_url()

# ---------------------------------------------------------------------------
# Engine và SessionFactory
# ---------------------------------------------------------------------------

_engine_kwargs: dict = {}

if DATABASE_URL.startswith("sqlite"):
    # SQLite cần check_same_thread=False khi dùng đa luồng
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    # Bật WAL mode để tránh lock khi nhiều reader/writer đồng thời
    _engine_kwargs["pool_pre_ping"] = True

    def _set_sqlite_pragmas(dbapi_conn, _connection_record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

else:
    _engine_kwargs["pool_pre_ping"] = True  # kiểm tra kết nối trước khi dùng
    _engine_kwargs["pool_size"] = 5
    _engine_kwargs["max_overflow"] = 10

engine = create_engine(DATABASE_URL, **_engine_kwargs)

# Áp dụng SQLite pragmas
if DATABASE_URL.startswith("sqlite"):
    event.listen(engine, "connect", _set_sqlite_pragmas)

SessionFactory = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,   # tránh lazy-load sau commit
)


# ---------------------------------------------------------------------------
# Context manager tiện dụng
# ---------------------------------------------------------------------------

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Trả về Session với auto-commit và auto-rollback khi có lỗi.

    Dùng:
        with get_session() as session:
            session.add(employee)
            session.commit()
    """
    session: Session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------

def create_all_tables() -> None:
    """Tạo tất cả bảng nếu chưa tồn tại (dùng lần đầu hoặc test)."""
    # Import để đảm bảo tất cả models được đăng ký với Base
    import app.database.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Đã tạo xong tất cả bảng database.")


def ping_database() -> bool:
    """Kiểm tra kết nối database. Trả True nếu OK."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Không kết nối được database: %s", exc)
        return False
