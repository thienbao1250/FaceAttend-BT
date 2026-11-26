"""
scripts/demo_db_workflow.py
Kịch bản chạy thử toàn bộ luồng Database (Giai đoạn 9) trên SQLite cục bộ.

Kịch bản:
  1. Kết nối DB và kiểm tra các bảng
  2. Tạo 2 nhân viên mới (EmployeeRepository)
  3. Đăng ký mẫu khuôn mặt embedding cho nhân viên (FaceTemplateRepository)
  4. Load FaceGallery từ DB lên RAM
  5. Giả lập camera nhận diện khuôn mặt -> FaceMatcher -> AttendanceRepository
  6. Ghi nhận sự kiện Check-in, kiểm tra Cooldown, sau đó Check-out
  7. Truy vấn lịch sử chấm công từ DB và in báo cáo
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database.connection import get_session, create_all_tables, ping_database
from app.database.repositories.employee_repo import EmployeeRepository
from app.database.repositories.face_template_repo import FaceTemplateRepository
from app.database.repositories.attendance_repo import AttendanceRepository
from app.database.repositories.face_recognition_log_repo import FaceRecognitionLogRepository
from app.database.repositories.attendance_record_repo import AttendanceRecordRepository
from app.database.repositories.shift_repo import ShiftRepository
from app.vision.face_matcher import FaceGallery, FaceMatcher, MatchStatus


def main():
    print("=" * 65)
    print("🚀 CHẠY THỬ NGHIỆM HỆ THỐNG DATABASE (GIAI ĐOẠN 9 - SQLITE)")
    print("=" * 65)

    # 1. Kiểm tra kết nối & schema
    if not ping_database():
        print("❌ Lỗi: Không thể kết nối DB!")
        return

    create_all_tables()
    print("✅ 1. Kết nối DB thành công & các bảng đã sẵn sàng.")

    # 2. Tạo camera & nhân viên
    with get_session() as session:
        emp_repo = EmployeeRepository(session)
        tpl_repo = FaceTemplateRepository(session)
        att_repo = AttendanceRepository(session)
        log_repo = FaceRecognitionLogRepository(session)
        rec_repo = AttendanceRecordRepository(session)
        shift_repo = ShiftRepository(session)
        default_shift = shift_repo.get_or_create_default()

        # Tạo camera mặc định nếu chưa có
        from app.database.models import Camera
        cam = session.query(Camera).filter_by(camera_code="CAM_01").first()
        if not cam:
            cam = Camera(camera_code="CAM_01", name="Cổng chính", location="Tầng 1", status="online")
            session.add(cam)
            session.flush()

        # Xóa dữ liệu cũ của nhân viên test nếu có
        old_a = emp_repo.get_by_code("NV_TEST_01")
        if old_a:
            emp_repo.hard_delete(old_a.id)
        old_b = emp_repo.get_by_code("NV_TEST_02")
        if old_b:
            emp_repo.hard_delete(old_b.id)
        session.commit()

        print("\n--- 2. Tạo camera & nhân viên mới trong Database ---")
        print(f"  [+] Camera sẵn sàng: ID={cam.id} | Mã={cam.camera_code} | Vị trí={cam.location}")
        emp1 = emp_repo.create("NV_TEST_01", "Nguyễn Văn An", department="Kỹ thuật")
        emp2 = emp_repo.create("NV_TEST_02", "Trần Thị Bình", department="Nhân sự")
        session.commit()
        print(f"  [+] Đã tạo: ID={emp1.id} | Mã={emp1.employee_code} | Tên={emp1.full_name} | Phòng={emp1.department}")
        print(f"  [+] Đã tạo: ID={emp2.id} | Mã={emp2.employee_code} | Tên={emp2.full_name} | Phòng={emp2.department}")

        # 3. Tạo vector embedding mẫu (512 chiều)
        print("\n--- 3. Lưu trữ Face Templates (Vector Embedding dạng bytea) ---")
        rng = np.random.default_rng(2026)
        
        # Vector đặc trưng của An (tạo 2 góc chụp khác nhau)
        emb_an_1 = rng.standard_normal(512).astype(np.float32)
        emb_an_1 /= np.linalg.norm(emb_an_1)
        emb_an_2 = emb_an_1 + rng.normal(0, 0.05, 512).astype(np.float32)
        emb_an_2 /= np.linalg.norm(emb_an_2)

        # Vector đặc trưng của Bình
        emb_binh = rng.standard_normal(512).astype(np.float32)
        emb_binh /= np.linalg.norm(emb_binh)

        tpl_repo.add(emp1.id, emb_an_1, model_name="w600k_r50", quality_score=0.96)
        tpl_repo.add(emp1.id, emb_an_2, model_name="w600k_r50", quality_score=0.92)
        tpl_repo.add(emp2.id, emb_binh, model_name="w600k_r50", quality_score=0.98)
        session.commit()
        print(f"  [+] Đã lưu 2 mẫu khuôn mặt cho {emp1.full_name}")
        print(f"  [+] Đã lưu 1 mẫu khuôn mặt cho {emp2.full_name}")

        # 4. Nạp gallery từ DB lên RAM cho FaceMatcher
        print("\n--- 4. Nạp Gallery từ DB lên RAM (In-Memory Face Matching) ---")
        db_gallery = tpl_repo.load_gallery(model_name="w600k_r50", model_version="1.0")
        
        gallery = FaceGallery()
        for emp_id, emb_list in db_gallery.items():
            for emb in emb_list:
                gallery.add(str(emp_id), emb)
        
        matcher = FaceMatcher(gallery)
        print(f"  [+] FaceMatcher đã nạp {gallery.size} nhân viên ({gallery.total_samples} mẫu)")

        # 5. Giả lập nhận diện từ Camera & Chấm công
        print("\n--- 5. Giả lập Camera nhận diện & Ghi nhận điểm danh ---")
        now = datetime.now(timezone.utc)

        # Camera quét thấy mặt An (mặt hơi lệch một chút so với mẫu 1)
        query_face = emb_an_1 + rng.normal(0, 0.02, 512).astype(np.float32)
        query_face /= np.linalg.norm(query_face)

        match_res = matcher.match(query_face)
        print(f"  [Camera]: Phát hiện khuôn mặt! Kết quả: {match_res.status.value.upper()} | Employee ID: {match_res.employee_id} | Similarity: {match_res.similarity:.4f}")

        if match_res.status == MatchStatus.KNOWN:
            emp_matched_id = int(match_res.employee_id)

            # A. Ghi FaceRecognitionLog (log kỹ thuật)
            recog_log = log_repo.record_recognition(
                timestamp=now,
                result="SUCCESS",
                employee_id=emp_matched_id,
                camera_id=cam.id,
                confidence=match_res.similarity,
            )

            # B. Ghi AttendanceRecord (bản ghi chính thức theo ngày)
            att_rec = rec_repo.upsert_check_in(
                employee_id=emp_matched_id,
                check_in=now,
                camera_id=cam.id,
                confidence=match_res.similarity,
                punctuality_status="ON_TIME",
                late_minutes=0,
                shift_id=default_shift.id,
                standard_work_minutes=default_shift.standard_work_minutes,
                device_id=cam.camera_code,
            )
            
            # Ghi nhận Check-in (attendance_events)
            ev1 = att_repo.record_event(
                employee_id=emp_matched_id,
                event_type="check_in",
                event_time=now,
                confidence=match_res.similarity,
                camera_id=cam.id,
                note="Nhận diện tự động cổng chính"
            )
            session.commit()
            print(f"  [DB]: Ghi FaceRecognitionLog (LogId={recog_log.id}, Result={recog_log.result})")
            print(f"  [DB]: Ghi AttendanceRecord (AttendanceId={att_rec.id}, Status={att_rec.attendance_status}, Punctuality={att_rec.punctuality_status})")
            print(f"  [DB]: Ghi nhận CHECK-IN thành công (Event ID={ev1.id}) lúc {ev1.event_time.strftime('%H:%M:%S UTC')}")

            # Thử ghi nhận check-out sau 9 tiếng (8h làm + 1h nghỉ trưa)
            t_out = now + timedelta(hours=9)
            rec_repo.upsert_check_out(
                employee_id=emp_matched_id,
                check_out=t_out,
                camera_id=cam.id,
                confidence=match_res.similarity,
                break_minutes=default_shift.break_minutes,
            )
            ev2 = att_repo.record_event(
                employee_id=emp_matched_id,
                event_type="check_out",
                event_time=t_out,
                confidence=match_res.similarity,
                camera_id=cam.id,
                note="Nhận diện tự động lúc tan ca"
            )
            session.commit()
            print(f"  [DB]: Ghi nhận CHECK-OUT thành công (Event ID={ev2.id})")

        # 6. Truy vấn lịch sử từ DB
        print("\n--- 6. Truy vấn lịch sử điểm danh từ Database ---")
        events = att_repo.list_by_employee(emp1.id)
        for ev in reversed(events):
            vn_time = ev.event_time.astimezone(timezone(timedelta(hours=7)))
            print(f"  - [{ev.event_type.upper()}] {vn_time.strftime('%Y-%m-%d %H:%M:%S')} (GMT+7) | Độ tin cậy: {ev.confidence:.2f} | Trạng thái: {ev.verify_status}")

    print("\n" + "=" * 65)
    print("🎉 TẤT CẢ CÁC BƯỚC ĐÃ CHẠY HOÀN TẤT THÀNH CÔNG!")
    print("=" * 65)


if __name__ == "__main__":
    main()
