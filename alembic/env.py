"""
alembic/env.py
Cấu hình Alembic để sử dụng SQLAlchemy engine và models của FaceAttend.

Hỗ trợ cả offline mode (tạo SQL script) và online mode (chạy migration trực tiếp).
DATABASE_URL đọc từ biến môi trường hoặc .env file nếu có python-dotenv.
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Thêm root project vào sys.path để import app.*
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Nạp .env nếu có python-dotenv (không bắt buộc)
try:
    from dotenv import load_dotenv
    env_file = ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Import Base và tất cả models để Alembic nhận biết schema
# ---------------------------------------------------------------------------
from app.database.connection import Base  # noqa: E402
import app.database.models  # noqa: F401, E402  — đảm bảo models được đăng ký với Base

# ---------------------------------------------------------------------------
# Alembic Config object (đọc alembic.ini)
# ---------------------------------------------------------------------------
config = context.config

# Đặt DATABASE_URL từ env vào alembic config (bắt buộc)
database_url = os.environ.get("DATABASE_URL", "").strip()
if not database_url:
    raise RuntimeError(
        "DATABASE_URL chưa được cấu hình trong .env hoặc biến môi trường."
    )
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql+psycopg2://", 1)
config.set_main_option("sqlalchemy.url", database_url)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# MetaData cho autogenerate
# ---------------------------------------------------------------------------
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline migration (tạo SQL script mà không cần kết nối DB)
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # cần cho SQLite (không hỗ trợ ALTER TABLE)
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migration (chạy trực tiếp trên DB đang kết nối)
# ---------------------------------------------------------------------------
def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # cần cho SQLite
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
