"""add_face_recognition_log_and_attendance_record

Revision ID: a2f9c3d1e8b7
Revises: 88f84ee5e4b6
Create Date: 2026-11-26 12:30:00.000000

Thêm hai bảng mới theo thiết kế nghiệp vụ Giai đoạn 10:

A. face_recognition_logs — Log kỹ thuật
   Lưu lại mỗi lần camera thực hiện nhận diện khuôn mặt (kể cả thất bại).
   Chính sách lưu trữ:
     - Log SUCCESS  : 30–90 ngày (cột expires_at)
     - Log FAILURE  : 7–30 ngày (cột expires_at)
     - Ảnh sự kiện : 7–30 ngày nếu thực sự cần (event_snapshot_path)

B. attendance_records — Bản ghi chính thức
   Chỉ lưu những sự kiện được hệ thống chấp nhận là chấm công.
   Phân cấp theo work_date (năm → tháng → ngày).
   Thêm trạng thái đúng/trễ giờ: punctuality_status, late_minutes.
   Chính sách lưu trữ: theo thời hạn pháp luật hoặc thỏa thuận.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a2f9c3d1e8b7'
down_revision: Union[str, None] = '88f84ee5e4b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ----------------------------------------------------------------
    # Bảng A: face_recognition_logs
    # ----------------------------------------------------------------
    op.create_table(
        'face_recognition_logs',
        sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False,
                  comment='LogId — tăng dần, VD: 10001'),
        sa.Column('employee_id', sa.Integer(), nullable=True,
                  comment='NULL nếu nhận diện thất bại (UNKNOWN / UNCERTAIN)'),
        sa.Column('camera_id', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False,
                  comment='Thời điểm nhận diện — lưu UTC (VD: 08:00:15)'),
        sa.Column('confidence', sa.Float(), nullable=True,
                  comment='Cosine similarity / confidence score (0.0–1.0), VD: 0.95'),
        sa.Column('result', sa.String(length=20), nullable=False,
                  comment='SUCCESS | FAILURE | UNCERTAIN'),
        sa.Column('failure_reason', sa.String(length=100), nullable=True,
                  comment='Lý do thất bại nếu result != SUCCESS'),
        sa.Column('event_snapshot_path', sa.String(length=500), nullable=True,
                  comment='Đường dẫn ảnh chụp sự kiện — xóa sau 7–30 ngày theo policy'),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True,
                  comment='Thời điểm hết hạn lưu trữ (scheduler sẽ xóa sau thời hạn này)'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('face_recognition_logs', schema=None) as batch_op:
        batch_op.create_index('ix_frl_timestamp', ['timestamp'], unique=False)
        batch_op.create_index('ix_frl_employee_time', ['employee_id', 'timestamp'], unique=False)
        batch_op.create_index('ix_frl_result', ['result'], unique=False)
        batch_op.create_index('ix_frl_expires_at', ['expires_at'], unique=False)

    # ----------------------------------------------------------------
    # Bảng B: attendance_records
    # ----------------------------------------------------------------
    op.create_table(
        'attendance_records',
        sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False,
                  comment='AttendanceId — tăng dần, VD: 10001'),
        sa.Column('employee_id', sa.Integer(), nullable=False,
                  comment='Không xóa cascade — giữ lịch sử dù nhân viên bị xóa'),
        sa.Column('work_date', sa.Date(), nullable=False,
                  comment='Ngày làm việc (YYYY-MM-DD). Phân cấp: năm → tháng → ngày'),

        # Check-in
        sa.Column('check_in', sa.DateTime(timezone=True), nullable=True,
                  comment='Thời điểm check-in — UTC (VD: 08:00:15)'),
        sa.Column('check_in_camera_id', sa.Integer(), nullable=True),
        sa.Column('check_in_confidence', sa.Float(), nullable=True,
                  comment='Confidence khi check-in (0.0–1.0)'),

        # Check-out
        sa.Column('check_out', sa.DateTime(timezone=True), nullable=True,
                  comment='Thời điểm check-out — UTC (VD: 17:05:20)'),
        sa.Column('check_out_camera_id', sa.Integer(), nullable=True),
        sa.Column('check_out_confidence', sa.Float(), nullable=True,
                  comment='Confidence khi check-out (0.0–1.0)'),

        # Trạng thái điểm danh
        sa.Column('attendance_status', sa.String(length=20), nullable=False,
                  comment='PRESENT | ABSENT | LATE | HALF_DAY | HOLIDAY | LEAVE'),
        sa.Column('punctuality_status', sa.String(length=20), nullable=False,
                  comment='Trạng thái đúng/trễ giờ: ON_TIME | LATE | EARLY_LEAVE | UNKNOWN'),
        sa.Column('late_minutes', sa.Integer(), nullable=True,
                  comment='Số phút đi trễ (0 nếu đúng giờ, NULL nếu không xác định)'),
        sa.Column('early_leave_minutes', sa.Integer(), nullable=True,
                  comment='Số phút về sớm'),
        sa.Column('work_minutes', sa.Integer(), nullable=True,
                  comment='Tổng số phút làm việc (check_out - check_in)'),

        # Ca làm việc
        sa.Column('shift_start', sa.DateTime(timezone=True), nullable=True,
                  comment='Giờ bắt đầu ca quy định — UTC'),
        sa.Column('shift_end', sa.DateTime(timezone=True), nullable=True,
                  comment='Giờ kết thúc ca quy định — UTC'),

        # Phương thức và xác minh — đảm bảo cấu trúc nghiệp vụ:
        # Employee ID | Thời gian | Device ID | Phương thức | Event Type | Verify Status | Result | Record ID
        sa.Column('verify_method', sa.String(length=50), nullable=False,
                  comment='Phương thức điểm danh: Face Recognition | Manual | Card'),
        sa.Column('verify_status', sa.String(length=20), nullable=False,
                  comment='Success | Failed | Disputed'),
        sa.Column('result', sa.String(length=20), nullable=False,
                  comment='Accepted | Rejected | Pending'),
        sa.Column('device_id', sa.String(length=50), nullable=True,
                  comment='ID thiết bị/camera ghi nhận'),
        sa.Column('record_source', sa.String(length=20), nullable=False,
                  comment='auto | manual | corrected'),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),

        sa.ForeignKeyConstraint(['check_in_camera_id'], ['cameras.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['check_out_camera_id'], ['cameras.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('employee_id', 'work_date', name='uq_attendance_emp_date'),
    )
    with op.batch_alter_table('attendance_records', schema=None) as batch_op:
        # Phân cấp theo năm/tháng/ngày
        batch_op.create_index('ix_attendance_records_work_date', ['work_date'], unique=False)
        batch_op.create_index('ix_attendance_records_emp_date', ['employee_id', 'work_date'], unique=False)
        batch_op.create_index('ix_attendance_records_status', ['attendance_status'], unique=False)
        batch_op.create_index('ix_attendance_records_punctuality', ['punctuality_status'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('attendance_records', schema=None) as batch_op:
        batch_op.drop_index('ix_attendance_records_punctuality')
        batch_op.drop_index('ix_attendance_records_status')
        batch_op.drop_index('ix_attendance_records_emp_date')
        batch_op.drop_index('ix_attendance_records_work_date')

    op.drop_table('attendance_records')

    with op.batch_alter_table('face_recognition_logs', schema=None) as batch_op:
        batch_op.drop_index('ix_frl_expires_at')
        batch_op.drop_index('ix_frl_result')
        batch_op.drop_index('ix_frl_employee_time')
        batch_op.drop_index('ix_frl_timestamp')

    op.drop_table('face_recognition_logs')
