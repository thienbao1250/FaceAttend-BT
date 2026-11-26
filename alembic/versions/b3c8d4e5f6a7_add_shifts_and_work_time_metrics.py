"""add_shifts_and_work_time_metrics

Revision ID: b3c8d4e5f6a7
Revises: a2f9c3d1e8b7
Create Date: 2026-11-26 13:00:00.000000

Bổ sung:
  1. Bảng `shifts` quản lý ca làm việc (giờ bắt đầu, kết thúc, nghỉ trưa, grace period, ca qua đêm).
  2. Các cột thời gian làm việc & đối soát trên `attendance_records`:
     - shift_id
     - total_presence_minutes, break_minutes, actual_work_minutes, standard_work_minutes
     - missing_minutes, extra_minutes, overtime_minutes, overtime_approved
     - needs_review, review_reason
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b3c8d4e5f6a7'
down_revision: Union[str, None] = 'a2f9c3d1e8b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'shifts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('shift_code', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('start_time', sa.Time(), nullable=False),
        sa.Column('end_time', sa.Time(), nullable=False),
        sa.Column('break_start', sa.Time(), nullable=True),
        sa.Column('break_end', sa.Time(), nullable=True),
        sa.Column('break_minutes', sa.Integer(), nullable=False, server_default='60'),
        sa.Column('standard_work_minutes', sa.Integer(), nullable=False, server_default='480'),
        sa.Column('grace_period_minutes', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('early_check_in_minutes', sa.Integer(), nullable=False, server_default='120'),
        sa.Column('late_check_out_minutes', sa.Integer(), nullable=False, server_default='240'),
        sa.Column('is_overnight', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('shift_code'),
    )

    with op.batch_alter_table('attendance_records', schema=None) as batch_op:
        batch_op.add_column(sa.Column('shift_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('total_presence_minutes', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('break_minutes', sa.Integer(), nullable=True, server_default='0'))
        batch_op.add_column(sa.Column('actual_work_minutes', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('standard_work_minutes', sa.Integer(), nullable=True, server_default='480'))
        batch_op.add_column(sa.Column('missing_minutes', sa.Integer(), nullable=True, server_default='0'))
        batch_op.add_column(sa.Column('extra_minutes', sa.Integer(), nullable=True, server_default='0'))
        batch_op.add_column(sa.Column('overtime_minutes', sa.Integer(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('overtime_approved', sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column('needs_review', sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column('review_reason', sa.String(length=200), nullable=True))
        batch_op.create_foreign_key(
            'fk_attendance_records_shift_id', 'shifts', ['shift_id'], ['id'], ondelete='SET NULL'
        )
        batch_op.create_index('ix_attendance_records_needs_review', ['needs_review'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('attendance_records', schema=None) as batch_op:
        batch_op.drop_index('ix_attendance_records_needs_review')
        batch_op.drop_constraint('fk_attendance_records_shift_id', type_='foreignkey')
        batch_op.drop_column('review_reason')
        batch_op.drop_column('needs_review')
        batch_op.drop_column('overtime_approved')
        batch_op.drop_column('overtime_minutes')
        batch_op.drop_column('extra_minutes')
        batch_op.drop_column('missing_minutes')
        batch_op.drop_column('standard_work_minutes')
        batch_op.drop_column('actual_work_minutes')
        batch_op.drop_column('break_minutes')
        batch_op.drop_column('total_presence_minutes')
        batch_op.drop_column('shift_id')

    op.drop_table('shifts')
