# FaceAttend AI — Workflow phát triển hệ thống điểm danh bằng nhận diện khuôn mặt

> **Mục tiêu:** Xây dựng một hệ thống dùng webcam/camera để phát hiện và nhận diện nhân viên, sau đó ghi nhận lượt vào/ra. Dự án được triển khai từng bước trên máy cá nhân trước; chỉ cân nhắc triển khai máy chủ sau khi phiên bản local hoạt động ổn định.
>
> **Nguyên tắc thực hiện:** Mỗi giai đoạn phải chạy được, có thể kiểm thử và có đầu ra rõ ràng trước khi chuyển sang giai đoạn tiếp theo. Không cài đồng loạt thư viện hoặc xây backend quá sớm.

---

## 1. Phạm vi và định hướng kỹ thuật

### 1.1. Chức năng mục tiêu

- Mở webcam hoặc camera IP và lấy khung hình.
- Phát hiện một hoặc nhiều khuôn mặt trong khung hình.
- Chuẩn hóa khuôn mặt và tạo đặc trưng (face embedding).
- So sánh đặc trưng với dữ liệu nhân viên đã đăng ký.
- Trả về một trong các kết quả: nhân viên đã biết, người chưa đăng ký, hoặc chưa đủ chất lượng để kết luận.
- Ghi nhận sự kiện vào/ra theo quy tắc chấm công.
- Cho phép quản lý nhân viên, dữ liệu đăng ký và lịch sử chấm công.
- Có nhật ký lỗi và cơ chế kiểm thử để dễ bảo trì.

### 1.2. Công nghệ định hướng

| Thành phần | Công nghệ định hướng | Vai trò |
|---|---|---|
| Ngôn ngữ | Python | Xử lý camera, AI và backend ban đầu |
| Camera / xử lý ảnh | OpenCV | Đọc webcam, lấy frame, hiển thị kết quả |
| Face Detection | SCRFD | Xác định vị trí khuôn mặt và các điểm mốc |
| Face Recognition | ArcFace | Chuyển khuôn mặt thành vector đặc trưng |
| AI Runtime | ONNX Runtime | Chạy mô hình ONNX; kiểm tra CPU trước, GPU sau |
| Backend API | FastAPI | Cung cấp API cho quản lý và chấm công |
| Database | PostgreSQL | Lưu nhân viên, dữ liệu đăng ký và sự kiện chấm công |
| Kiểm thử | pytest | Kiểm thử logic và API |
| Đóng gói | Docker (giai đoạn sau) | Chuẩn hóa môi trường chạy/deploy |

**Lưu ý lựa chọn công nghệ:**
- YOLO không bắt buộc cho bài toán nhận diện danh tính. YOLO thường dùng để phát hiện đối tượng; phát hiện khuôn mặt và nhận diện danh tính là hai nhiệm vụ riêng.
- Frigate phù hợp hơn khi cần hệ thống NVR/quản lý camera giám sát. Với một thiết bị điểm danh chuyên dụng, có thể bắt đầu trực tiếp bằng OpenCV và pipeline nhận diện.
- gRPC và Docker chưa cần ở giai đoạn đầu. Chỉ bổ sung khi có nhu cầu tách dịch vụ, giao tiếp giữa nhiều thiết bị hoặc triển khai nhiều môi trường.
- Tên mô hình, phiên bản thư viện, ngưỡng nhận diện và cấu hình GPU phải được xác minh theo source code hiện tại và môi trường thực tế; không chọn ngẫu nhiên.

### 1.3. Giới hạn phần cứng cần lưu ý

Máy hiện tại có NVIDIA RTX 3050 4 GB VRAM và 8 GB RAM. Vì vậy:
- Ưu tiên mô hình gọn và xử lý theo từng frame hoặc tốc độ vừa phải.
- Đo mức sử dụng RAM/VRAM trước khi tăng độ phân giải, số camera hoặc số luồng xử lý.
- Chạy được trên CPU trước là mốc cơ sở; sau đó mới cấu hình ONNX Runtime GPU và so sánh hiệu năng.
- Không đồng nghĩa driver NVIDIA hiển thị phiên bản CUDA là máy đã cài CUDA Toolkit hoặc đã sẵn sàng cho mọi gói GPU.

---

## 2. Sơ đồ luồng xử lý tổng thể

```text
Webcam / Camera IP
        |
        v
[1. Camera Capture]
        |
        v
[2. Frame Pre-processing]
        |
        v
[3. Face Detection - SCRFD]
        |
        v
[4. Face Alignment / Quality Check]
        |
        v
[5. Face Embedding - ArcFace]
        |
        v
[6. Face Matching]
        |
        +------------------------------+
        |                              |
        v                              v
 Nhân viên đã biết              Chưa biết / Không chắc chắn
        |                              |
        v                              v
[7. Attendance Rules]          Hiển thị cần đăng ký / xác minh
        |
        v
[8. Attendance Event]
        |
        v
[9. Database + API + Admin UI]
```

Các bước 1–6 là pipeline thị giác máy tính/AI. Bước 7–9 là nghiệp vụ điểm danh và phần mềm quản lý. Không nên gộp tất cả vào một hàm Python lớn.

---

## 3. Workflow chi tiết theo từng giai đoạn

## Giai đoạn 0 — Khảo sát repository và môi trường hiện có

**Mục tiêu:** Hiểu source code đã clone trước khi thêm thư viện hoặc thay đổi cấu trúc.

**Các việc thực hiện:**
1. Xác nhận đang ở đúng thư mục dự án và `.venv` đã được kích hoạt.
2. Xem danh sách thư mục/tệp, đặc biệt là `README.md`, `requirements.txt`, `pyproject.toml`, file cấu hình và mã nguồn hiện có.
3. Kiểm tra phiên bản Python, pip, NumPy, OpenCV và đường dẫn Python đang chạy.
4. Xác định repository hiện có là project khung, source tham khảo hay đã chứa pipeline AI.
5. Ghi lại các thư viện đã cài và lỗi (nếu có); không cài trùng hoặc nâng cấp hàng loạt khi chưa biết yêu cầu dự án.

**Đầu ra:**
- Bản mô tả cấu trúc source hiện tại.
- Danh sách thư viện/phụ thuộc hiện có.
- Quyết định giữ, điều chỉnh hay bổ sung cấu trúc dự án.

**Tiêu chí hoàn thành:** Biết được file chạy chính, cấu hình phụ thuộc và cách chạy thử project mà không làm hỏng môi trường.

---

## Giai đoạn 1 — Camera Capture: lấy hình từ camera

**Mục tiêu:** Ứng dụng mở được webcam, đọc frame liên tục và đóng camera an toàn.

**Các việc thực hiện:**
1. Dùng OpenCV mở webcam bằng camera index phù hợp.
2. Kiểm tra trạng thái `isOpened()` và xử lý trường hợp camera không khả dụng.
3. Đọc frame trong vòng lặp, hiển thị cửa sổ preview.
4. Cho phép thoát bằng phím; luôn giải phóng camera và cửa sổ khi kết thúc.
5. Kiểm tra kích thước frame, FPS và độ ổn định trong vài phút.
6. Tách phần camera thành module riêng, không đưa logic nhận diện vào module này.

**Đầu ra:** Module camera có thể mở, đọc, cung cấp frame và đóng thiết bị.

**Tiêu chí hoàn thành:** Webcam chạy ổn định, không bị treo, có thông báo rõ khi không mở được camera.

**Kiểm thử:** Camera bị ngắt; camera index sai; người dùng thoát ứng dụng; frame trả về rỗng.

---

## Giai đoạn 2 — Frame Pre-processing: chuẩn bị hình ảnh

**Mục tiêu:** Chuẩn hóa frame trước khi đưa vào mô hình AI.

**Các việc thực hiện:**
1. Xác định định dạng màu mà OpenCV trả về và định dạng mà mô hình yêu cầu.
2. Chọn kích thước xử lý phù hợp với mô hình và phần cứng.
3. Thực hiện resize, chuyển màu, chuẩn hóa pixel theo đúng tài liệu của mô hình.
4. Giữ tỷ lệ ảnh hoặc tính toán padding phù hợp để tọa độ khuôn mặt không bị lệch.
5. Không thay đổi thứ tự kênh màu hay chuẩn hóa theo suy đoán; phải dựa vào cấu hình model.

**Đầu ra:** Hàm tiền xử lý nhận frame và trả tensor/ảnh đúng định dạng đầu vào của model.

**Tiêu chí hoàn thành:** Kích thước, dtype, thứ tự kênh và miền giá trị đầu vào khớp với model.

---

## Giai đoạn 3 — Face Detection: phát hiện khuôn mặt bằng SCRFD

**Mục tiêu:** Tìm được vị trí khuôn mặt trong frame, chưa cần xác định đó là ai.

**Các việc thực hiện:**
1. Xác định model SCRFD được dùng, định dạng ONNX và giấy phép sử dụng.
2. Đọc input/output của model và cấu hình preprocessing/postprocessing.
3. Khởi tạo ONNX Runtime session trên CPU trước để kiểm chứng pipeline.
4. Chạy model trên frame đã tiền xử lý.
5. Giải mã bounding box, confidence và facial landmarks (nếu model cung cấp).
6. Áp dụng ngưỡng confidence và Non-Maximum Suppression theo cấu hình model.
7. Vẽ bounding box và landmarks lên frame để kiểm tra trực quan.

**Đầu ra:** Danh sách khuôn mặt phát hiện được trong mỗi frame, kèm tọa độ và confidence.

**Tiêu chí hoàn thành:** Phát hiện đúng vị trí khuôn mặt trên ảnh thử nghiệm; không vẽ box sai do nhầm tọa độ hoặc resize.

**Kiểm thử:** Không có mặt; một mặt; nhiều mặt; mặt ở rìa ảnh; ảnh tối; mặt nghiêng; ảnh bị mờ.

**Chưa làm ở bước này:** Không gán tên nhân viên, không ghi chấm công.

---

## Giai đoạn 4 — Face Alignment và kiểm tra chất lượng khuôn mặt

**Mục tiêu:** Cắt và căn chỉnh khuôn mặt nhất quán trước khi trích xuất đặc trưng.

**Các việc thực hiện:**
1. Dùng bounding box để lấy vùng khuôn mặt.
2. Dùng facial landmarks để căn chỉnh khuôn mặt theo chuẩn đầu vào của ArcFace.
3. Resize crop đúng kích thước model yêu cầu.
4. Kiểm tra chất lượng cơ bản: mặt quá nhỏ, bị mờ, thiếu sáng, bị che khuất hoặc góc quá nghiêng.
5. Nếu chất lượng không đạt, yêu cầu người dùng đứng gần/nhìn về camera thay vì cố nhận diện.
6. Lưu ảnh debug trong môi trường phát triển khi cần; tránh lưu ảnh sinh trắc học thật nếu không cần thiết.

**Đầu ra:** Khuôn mặt đã căn chỉnh và trạng thái chất lượng (đạt/không đạt).

**Tiêu chí hoàn thành:** Crop khuôn mặt đúng, ổn định và đưa vào model theo đúng kích thước/định dạng.

---

## Giai đoạn 5 — Face Embedding: trích xuất đặc trưng bằng ArcFace

**Mục tiêu:** Chuyển một khuôn mặt thành vector số đại diện cho đặc trưng khuôn mặt.

**Các việc thực hiện:**
1. Chọn model ArcFace tương thích với pipeline alignment.
2. Kiểm tra license, nguồn model, input size, normalization và output dimension.
3. Tạo ONNX Runtime session; chạy CPU trước.
4. Đưa crop đã căn chỉnh vào model để lấy embedding.
5. Chuẩn hóa vector theo đúng cách model yêu cầu (ví dụ L2 normalization nếu pipeline quy định).
6. Kiểm tra vector có đúng kích thước, không chứa NaN/Inf và không phải vector rỗng.
7. Tách chức năng embedding khỏi camera và database.

**Đầu ra:** Một vector đặc trưng cho mỗi khuôn mặt đạt chất lượng.

**Tiêu chí hoàn thành:** Cùng một người trong các ảnh khác nhau tạo embedding có độ tương đồng cao hơn so với hai người khác nhau trên bộ ảnh thử nghiệm.

**Lưu ý:** Embedding là dữ liệu sinh trắc học nhạy cảm; cần giới hạn quyền truy cập, mã hóa khi lưu/truyền phù hợp và chỉ thu thập khi có căn cứ, thông báo/đồng ý hợp lệ.

---

## Giai đoạn 6 — Face Matching: so khớp danh tính

**Mục tiêu:** So sánh embedding hiện tại với embedding đã đăng ký để xác định danh tính.

**Các việc thực hiện:**
1. Tạo bộ dữ liệu thử nghiệm có nhiều ảnh của mỗi người và ảnh của người không đăng ký.
2. Tính độ tương đồng giữa embedding hiện tại và embedding lưu trữ (thường dùng cosine similarity nếu embedding đã chuẩn hóa).
3. Xác định cách so khớp: so với từng mẫu, trung bình nhiều mẫu hay chiến lược khác phù hợp dữ liệu.
4. Hiệu chỉnh ngưỡng nhận diện trên tập validation riêng; không lấy một ngưỡng trên mạng rồi mặc định là đúng.
5. Thiết kế ba trạng thái: `known` (đủ tin cậy), `unknown` (không khớp), `uncertain` (gần ngưỡng/thiếu chất lượng).
6. Kiểm tra trường hợp hai người có điểm tương đồng gần nhau; không tự động gán người có điểm cao nhất nếu chưa đủ chắc chắn.

**Đầu ra:** Kết quả so khớp gồm trạng thái, mã nhân viên (nếu đủ tin cậy), điểm tương đồng và thông tin debug cần thiết.

**Tiêu chí hoàn thành:** Bộ thử nghiệm phân biệt được người đã đăng ký và người lạ với mức sai số được đo lường, không chỉ dựa trên vài lần chạy thành công.

**Chỉ số nên ghi nhận:** False Accept Rate (FAR), False Reject Rate (FRR), tỷ lệ không kết luận được và độ trễ nhận diện.

---

## Giai đoạn 7 — Employee Enrollment: đăng ký nhân viên

**Mục tiêu:** Tạo quy trình đăng ký dữ liệu khuôn mặt có kiểm soát.

**Các việc thực hiện:**
1. Nhập thông tin cơ bản và mã nhân viên duy nhất.
2. Thông báo mục đích sử dụng dữ liệu khuôn mặt và thực hiện quy trình thông báo/đồng ý, quyền rút lại và xóa dữ liệu theo chính sách áp dụng.
3. Hướng dẫn nhân viên nhìn camera và thu một số ảnh ở điều kiện/góc nhìn phù hợp.
4. Phát hiện, kiểm tra chất lượng, căn chỉnh và tạo embedding cho từng ảnh.
5. Loại mẫu lỗi hoặc chất lượng thấp; kiểm tra các mẫu có nhất quán không.
6. Lưu embedding gắn với mã nhân viên; chỉ lưu ảnh gốc nếu thật sự cần và đã có chính sách bảo vệ rõ ràng.
7. Cho phép cập nhật, vô hiệu hóa hoặc xóa hồ sơ và toàn bộ embedding liên quan.

**Đầu ra:** Hồ sơ nhân viên có các mẫu embedding đã xác thực.

**Tiêu chí hoàn thành:** Nhân viên đăng ký được; dữ liệu lỗi bị từ chối; có thể cập nhật/xóa hồ sơ theo quyền được cấp.

---

## Giai đoạn 8 — Attendance Rules: logic chấm công

**Mục tiêu:** Chuyển kết quả nhận diện thành sự kiện chấm công hợp lệ.

**Các việc thực hiện:**
1. Định nghĩa kiểu sự kiện: check-in, check-out hoặc sự kiện theo ca.
2. Xác định quy tắc chống ghi nhận liên tục khi một người đứng trước camera nhiều frame.
3. Dùng tracking hoặc cooldown theo nhân viên/camera để tránh ghi nhiều sự kiện giống nhau trong thời gian ngắn.
4. Quy định cách xử lý khi nhân viên xuất hiện lại, đổi camera, hoặc nhận diện không chắc chắn.
5. Xử lý trường hợp đi muộn, về sớm, thiếu lượt vào/ra và làm nhiều ca theo chính sách nghiệp vụ.
6. Dùng thời gian từ máy chủ có timezone rõ ràng; lưu timestamp nhất quán và hiển thị theo múi giờ nghiệp vụ.
7. Không tự suy ra check-in/check-out chỉ từ một lần nhận diện nếu quy tắc ca làm chưa được xác định.

**Đầu ra:** Hàm/service quyết định có tạo sự kiện chấm công hay không và loại sự kiện nào.

**Tiêu chí hoàn thành:** Một lượt nhận diện liên tục không tạo hàng loạt bản ghi; quy tắc vào/ra có test cho các tình huống biên.

---

## Giai đoạn 9 — Database: thiết kế và lưu trữ

**Mục tiêu:** Lưu dữ liệu có cấu trúc, tránh trùng lặp và có thể truy vấn lịch sử.

**Các bảng gợi ý (sẽ tinh chỉnh khi chốt nghiệp vụ):**

| Bảng | Dữ liệu chính |
|---|---|
| `employees` | Mã nhân viên, họ tên, trạng thái, thông tin cơ bản |
| `face_templates` | Employee ID, embedding, model version, thời điểm đăng ký |
| `attendance_events` | Employee ID, loại sự kiện, thời điểm, camera ID, trạng thái xác minh |
| `cameras` | Tên camera, vị trí, trạng thái và cấu hình không nhạy cảm |
| `users` / `roles` | Tài khoản quản trị và quyền truy cập |
| `audit_logs` | Ai đã xem/sửa/xóa dữ liệu quan trọng và thời điểm thực hiện |

**Các việc thực hiện:**
1. Vẽ ERD trước khi tạo bảng.
2. Xác định khóa chính, khóa ngoại, unique constraints và index.
3. Chọn kiểu lưu embedding phù hợp; version hóa model để biết embedding được tạo bằng phiên bản nào.
4. Dùng migration để quản lý thay đổi schema.
5. Thiết lập backup/restore và quyền truy cập tối thiểu.
6. Không ghi embedding, ảnh khuôn mặt hoặc thông tin nhạy cảm vào log thông thường.

**Đầu ra:** Database schema và migration đầu tiên.

**Tiêu chí hoàn thành:** Lưu, truy vấn, cập nhật và xóa dữ liệu đúng ràng buộc; có thể khôi phục bản backup thử nghiệm.

---

## Giai đoạn 10 — Backend API: FastAPI

**Mục tiêu:** Cung cấp API để giao diện quản lý và dịch vụ camera tương tác với hệ thống.

**Nhóm endpoint dự kiến:**

| Nhóm | Ví dụ chức năng |
|---|---|
| Employees | Tạo, xem, cập nhật, vô hiệu hóa nhân viên |
| Enrollment | Bắt đầu đăng ký, gửi mẫu, hoàn tất đăng ký |
| Recognition | Gửi yêu cầu nhận diện hoặc nhận kết quả từ service camera |
| Attendance | Tạo sự kiện hợp lệ, xem lịch sử, lọc theo ngày/nhân viên |
| Cameras | Quản lý camera và trạng thái thiết bị |
| Admin/Auth | Đăng nhập, phân quyền, quản lý tài khoản |

**Các việc thực hiện:**
1. Tạo FastAPI app và cấu hình môi trường.
2. Tách router, schema, service, repository và model database.
3. Kiểm tra dữ liệu đầu vào bằng schema.
4. Thêm xác thực và phân quyền trước khi mở API quản trị.
5. Xử lý lỗi thống nhất và ghi log không chứa dữ liệu sinh trắc học.
6. Viết API tests bằng pytest/TestClient.
7. Tài liệu hóa API bằng OpenAPI do FastAPI cung cấp.

**Đầu ra:** API có thể tạo/xem nhân viên, đăng ký, nhận sự kiện điểm danh và xem lịch sử.

**Tiêu chí hoàn thành:** API chạy độc lập, có kiểm thử và không cho người không đủ quyền truy cập chức năng quản trị.

---

## Giai đoạn 11 — Giao diện quản trị

**Mục tiêu:** Cho quản trị viên thao tác với hệ thống mà không cần sửa dữ liệu thủ công.

**Màn hình dự kiến:**
- Đăng nhập.
- Dashboard tổng quan lượt chấm công.
- Danh sách và chi tiết nhân viên.
- Quy trình đăng ký/cập nhật khuôn mặt.
- Lịch sử chấm công, tìm kiếm và lọc.
- Quản lý camera, trạng thái kết nối.
- Quản lý tài khoản, vai trò và nhật ký thao tác.

**Các việc thực hiện:**
1. Chọn công nghệ frontend sau khi API ổn định.
2. Dựng wireframe các màn hình chính.
3. Tích hợp API và xử lý trạng thái loading/error/empty.
4. Ẩn chức năng theo quyền người dùng.
5. Kiểm tra trên kích thước màn hình mục tiêu.

**Đầu ra:** Giao diện quản trị dùng được cho các nghiệp vụ chính.

**Tiêu chí hoàn thành:** Quản trị viên có thể quản lý nhân viên, đăng ký khuôn mặt và xem lịch sử điểm danh qua giao diện.

---

## Giai đoạn 12 — Liveness / Presentation Attack Detection

**Mục tiêu:** Giảm nguy cơ đánh lừa hệ thống bằng ảnh in, ảnh trên điện thoại hoặc video phát lại.

**Các việc thực hiện:**
1. Xác định mô hình đe dọa: ảnh in, video trên màn hình, video phát lại và các kiểu giả mạo khác.
2. Thiết kế thử nghiệm riêng cho từng loại tấn công.
3. Đánh giá giải pháp phù hợp: kiểm tra chuyển động/challenge-response hoặc mô hình PAD chuyên dụng.
4. Tích hợp PAD trước khi cho phép tạo sự kiện chấm công.
5. Đo FAR/FRR và tỷ lệ từ chối nhầm trong điều kiện thực tế.

**Đầu ra:** Trạng thái liveness/PAD và chính sách xử lý khi không đạt.

**Tiêu chí hoàn thành:** Có báo cáo thử nghiệm với ảnh/video giả mạo; không tuyên bố hệ thống chống giả mạo tuyệt đối.

**Lưu ý:** Nhận diện khuôn mặt đơn thuần không chứng minh người thật đang đứng trước camera.

---

## Giai đoạn 13 — GPU Acceleration và tối ưu hiệu năng

**Mục tiêu:** Tận dụng RTX 3050 khi môi trường tương thích và có lợi ích đo được.

**Các việc thực hiện:**
1. Hoàn tất pipeline CPU trước và lưu kết quả benchmark làm mốc so sánh.
2. Xác minh phiên bản ONNX Runtime GPU tương thích với hệ điều hành, Python, CUDA và cuDNN cần thiết.
3. Cài đúng runtime và kiểm tra provider thực tế được ONNX Runtime nhận diện.
4. Chạy cùng dữ liệu trên CPU và GPU, so sánh độ trễ, FPS, RAM và VRAM.
5. Kiểm tra bộ nhớ khi chạy lâu; xử lý giải phóng session/frame và giới hạn hàng đợi.
6. Chỉ giữ GPU nếu có cải thiện ổn định, không làm môi trường khó bảo trì hơn mức cần thiết.

**Đầu ra:** Bảng benchmark CPU/GPU và cấu hình chạy được ghi lại.

**Tiêu chí hoàn thành:** Xác nhận được execution provider thực tế và hiệu năng có cải thiện trong điều kiện thử nghiệm.

---

## Giai đoạn 14 — Logging, Monitoring và xử lý lỗi

**Mục tiêu:** Có thể phát hiện và điều tra sự cố mà không làm lộ dữ liệu cá nhân.

**Các việc thực hiện:**
1. Ghi log trạng thái camera, lỗi model, thời gian xử lý, lỗi API và lỗi database.
2. Dùng mã sự kiện/request ID để liên kết log giữa các module.
3. Theo dõi camera mất kết nối, FPS giảm, API lỗi và dung lượng lưu trữ.
4. Quy định thời gian lưu log và quyền truy cập log.
5. Không lưu ảnh khuôn mặt/embedding trong log; dữ liệu debug phải có kiểm soát và thời hạn xóa.

**Đầu ra:** Log có cấu trúc và hướng dẫn xử lý các lỗi thường gặp.

**Tiêu chí hoàn thành:** Có thể xác định module gây lỗi từ log mà không cần thu thập dữ liệu sinh trắc học không cần thiết.

---

## Giai đoạn 15 — Kiểm thử tổng thể và đánh giá

**Mục tiêu:** Xác minh cả độ chính xác AI lẫn tính đúng đắn của nghiệp vụ điểm danh.

**Các lớp kiểm thử:**

| Lớp | Nội dung |
|---|---|
| Unit test | Preprocessing, matching, threshold, attendance rules |
| Model test | Detection, alignment, embedding và kết quả trên bộ dữ liệu đánh giá |
| Integration test | Camera → AI pipeline → service → database |
| API test | Request hợp lệ/không hợp lệ, xác thực, phân quyền |
| End-to-end test | Đăng ký nhân viên → nhận diện → tạo sự kiện → tra cứu lịch sử |
| Performance test | Độ trễ, FPS, RAM/VRAM, chạy liên tục |
| Security test | Quyền truy cập, dữ liệu nhạy cảm, đăng nhập và audit log |
| Reliability test | Rút camera, mất mạng, restart service, database tạm thời không khả dụng |

**Các tình huống bắt buộc:**
- Nhân viên đã đăng ký đứng trước camera.
- Người chưa đăng ký đứng trước camera.
- Hai hoặc nhiều người cùng xuất hiện.
- Mặt bị che, thiếu sáng, nghiêng hoặc quá xa.
- Ảnh/video được đưa trước camera.
- Một người đứng trước camera lâu và không bị ghi nhận lặp liên tục.
- Camera hoặc database bị mất kết nối rồi hoạt động trở lại.

**Đầu ra:** Test report, lỗi đã biết và quyết định có đủ điều kiện chuyển sang triển khai thử hay chưa.

---

## Giai đoạn 16 — Đóng gói và triển khai thử nghiệm

**Mục tiêu:** Chạy hệ thống ổn định trên một máy/camera thử nghiệm trước khi mở rộng.

**Các việc thực hiện:**
1. Chốt file cấu hình môi trường và quản lý secret ngoài source code.
2. Viết hướng dẫn cài đặt, chạy, dừng và khôi phục hệ thống.
3. Có thể dùng Docker cho backend/database khi đã hiểu rõ các service và phụ thuộc.
4. Thử nghiệm với một camera và nhóm người tham gia tự nguyện/được thông báo đầy đủ.
5. Theo dõi tỷ lệ nhận diện sai, bỏ sót, lỗi ghi công và phản hồi người dùng.
6. Có cách ghi nhận/chỉnh sửa thủ công khi AI không kết luận được.
7. Chỉ mở rộng số camera khi đã đo được tải hệ thống và năng lực xử lý.

**Đầu ra:** Bản thử nghiệm có tài liệu triển khai, backup, rollback và quy trình xử lý sự cố.

**Tiêu chí hoàn thành:** Hệ thống chạy liên tục trong thời gian thử nghiệm đã định, có thể khôi phục và có phương án xử lý trường hợp nhận diện sai.

---

## 4. Cấu trúc source code mục tiêu (đề xuất)

Đây là cấu trúc tham khảo, **không phải yêu cầu tạo toàn bộ thư mục ngay lập tức**. Sau Giai đoạn 0 sẽ điều chỉnh theo repository đã clone.

```text
FaceAttend-BT/
├── app/
│   ├── camera/
│   │   └── camera_capture.py
│   ├── vision/
│   │   ├── preprocessing.py
│   │   ├── face_detector.py
│   │   ├── face_alignment.py
│   │   ├── face_quality.py
│   │   └── face_recognizer.py
│   ├── attendance/
│   │   └── attendance_service.py
│   ├── enrollment/
│   │   └── enrollment_service.py
│   ├── api/
│   │   ├── main.py
│   │   └── routers/
│   ├── database/
│   │   ├── models.py
│   │   └── session.py
│   └── core/
│       ├── config.py
│       └── logging.py
├── models/                 # Model weights; kiểm tra license trước khi dùng
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/
├── docs/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

Không commit `.venv/`, file `.env` thật, ảnh khuôn mặt thật, embedding thật hoặc dữ liệu chấm công cá nhân lên Git.

---

## 5. Thứ tự triển khai thực tế

Để tránh làm nhiều phần cùng lúc, thứ tự làm việc được chốt như sau:

| Thứ tự | Công việc | Trạng thái ban đầu |
|---:|---|---|
| 0 | Khảo sát repository và môi trường | Đang cần kiểm tra source hiện tại |
| 1 | Webcam capture và preview | Đã chạy test camera thành công |
| 2 | Tiền xử lý frame | Chưa thực hiện |
| 3 | SCRFD phát hiện khuôn mặt | Chưa thực hiện |
| 4 | Căn chỉnh và kiểm tra chất lượng mặt | Chưa thực hiện |
| 5 | ArcFace tạo embedding | Chưa thực hiện |
| 6 | So khớp danh tính và unknown/uncertain | Chưa thực hiện |
| 7 | Đăng ký nhân viên | ✅ Hoàn thành |
| 8 | Quy tắc chấm công | Chưa thực hiện |
| 9 | Database | Chưa thực hiện |
| 10 | FastAPI | Chưa thực hiện |
| 11 | Giao diện quản trị | Chưa thực hiện |
| 12 | Liveness/PAD | Chưa thực hiện |
| 13 | Tối ưu GPU | Chưa thực hiện |
| 14–16 | Logging, kiểm thử tổng thể, triển khai thử | Chưa thực hiện |

Có thể điều chỉnh thứ tự một số hạng mục khi repository hiện tại đã có sẵn module tương ứng. Tuy nhiên, không nên chuyển sang nghiệp vụ chấm công trước khi phát hiện và nhận diện khuôn mặt đã được kiểm thử riêng.

---

## 6. Quy tắc làm việc cho từng buổi

Mỗi lần chỉ nên tập trung vào một mục tiêu nhỏ:

1. **Nêu mục tiêu:** Hôm nay cần làm được điều gì?
2. **Giải thích kiến thức:** Module này nhận đầu vào gì, xử lý gì và trả đầu ra gì?
3. **Khảo sát source hiện tại:** Tận dụng code đang có, tránh viết lại không cần thiết.
4. **Thực hiện thay đổi nhỏ:** Không sửa nhiều module cùng lúc.
5. **Chạy thử:** Có lệnh chạy và kết quả mong đợi rõ ràng.
6. **Kiểm thử lỗi:** Thử ít nhất một trường hợp không lý tưởng.
7. **Ghi nhận:** Cập nhật README/worklog, phiên bản thư viện và vấn đề còn tồn tại.
8. **Chỉ chuyển bước khi đạt tiêu chí hoàn thành.**

### Quy ước khi sửa code

- Không ghi đè hoặc xóa source có sẵn khi chưa đọc và hiểu.
- Không cài thư viện chỉ vì thấy nó xuất hiện trong một hướng dẫn khác.
- Không đưa token, mật khẩu, connection string hoặc dữ liệu cá nhân lên Git.
- Tách cấu hình khỏi code; dùng file `.env` cục bộ và `.env.example` không chứa secret.
- Mỗi module có trách nhiệm rõ ràng, dễ kiểm thử độc lập.
- Khi model hoặc preprocessing thay đổi, phải version hóa và đánh giá lại embedding/ngưỡng nhận diện.

---

## 7. Quyền riêng tư và an toàn dữ liệu

Dữ liệu khuôn mặt và embedding có thể được dùng để nhận diện một cá nhân, vì vậy cần được bảo vệ như dữ liệu sinh trắc học nhạy cảm.

- Chỉ thu thập cho mục đích đã thông báo và có căn cứ xử lý phù hợp.
- Hạn chế người có quyền đăng ký, xem, xuất và xóa dữ liệu.
- Chỉ lưu dữ liệu thực sự cần; đặt thời hạn lưu và quy trình xóa.
- Mã hóa dữ liệu khi lưu/truyền phù hợp với mô hình triển khai.
- Có nhật ký truy cập và thay đổi dữ liệu quan trọng.
- Có phương án chấm công thủ công khi nhận diện thất bại hoặc người lao động không thể sử dụng hệ thống.
- Không coi nhận diện khuôn mặt là bằng chứng tuyệt đối; cần quy trình xác minh và xử lý khiếu nại.
- Trước khi dùng trong môi trường lao động thực tế, cần rà soát quy định pháp luật Việt Nam hiện hành về dữ liệu cá nhân, dữ liệu sinh trắc học và quan hệ lao động.

---

## 8. Việc cần làm ngay tiếp theo

Camera đã chạy thử thành công. Bước kế tiếp là **Giai đoạn 0 — khảo sát repository**, không phải cài thêm AI runtime ngay.

Trong thư mục `E:\Hoc_AI\FaceAttend-BT`, khi `.venv` đã kích hoạt, chạy:

```powershell
Get-ChildItem -Force
```

Sau đó:

```powershell
if (Test-Path README.md) { Get-Content README.md }
if (Test-Path requirements.txt) { Get-Content requirements.txt }
```

Gửi cấu trúc thư mục và nội dung các file có sẵn. Từ đó mới quyết định nên giữ kiến trúc repository hiện tại hay điều chỉnh, rồi tiến hành **Giai đoạn 2 — tiền xử lý frame** sau khi xác nhận webcam module đang có.

---

## 9. Nhật ký tiến độ

| Ngày | Giai đoạn | Kết quả | Ghi chú |
|---|---|---|---|
| 2026-11-12 | Môi trường Python | Đã khởi tạo và kích hoạt `.venv` theo trao đổi | Cần xác nhận thông tin package hiện tại trong repository |
| 2026-11-12 | Camera Capture | Đã chạy test webcam thành công | Cần kiểm tra source để biết code test đang nằm ở đâu |
| 2026-11-13 | Giai đoạn 2 — Preprocessing | Hoàn thành `preprocessing.py`: letterbox, normalize, NCHW tensor | Tương thích SCRFD + ArcFace |
| 2026-11-13 | Giai đoạn 3 — Face Detection | Hoàn thành `face_detector.py`: SCRFD ONNX, NMS, landmarks | `det_10g.onnx`; script `run_face_detection.py` |
| 2026-11-13 | Giai đoạn 4 — Alignment + Quality | Hoàn thành `face_alignment.py` + `face_quality.py` | 47 unit tests passed; script `run_face_alignment.py` |
| 2026-11-13 | Giai đoạn 5 — Face Embedding (ArcFace) | Hoàn thành `face_recognizer.py`; 42 unit tests passed | `w600k_r50.onnx`; embedding (512,) L2-norm; script `run_face_embedding.py` |
| 2026-11-13 | Giai đoạn 6 — Face Matching | Hoàn thành `face_matcher.py`: FaceGallery + FaceMatcher + evaluate (FAR/FRR) | 3 trạng thái KNOWN/UNKNOWN/UNCERTAIN; script `run_face_matching.py` |
| 2026-11-13 | Giai đoạn 7 — Employee Enrollment | Hoàn thành `enrollment_service.py`; 40 unit tests passed | Luồng: create_profile → add_sample → finalize; set_status/delete; script `run_enrollment.py` |

