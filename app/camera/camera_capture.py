"""
app/camera/camera_capture.py
Module quản lý camera: mở, đọc frame, cung cấp frame và đóng an toàn.
Không chứa bất kỳ logic nhận diện hay AI nào.
"""

import logging
import cv2

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Lớp quản lý vòng đời của camera: mở, đọc frame và đóng.

    Tham số:
        camera_index (int): Chỉ số camera (mặc định 0 = webcam mặc định).
        backend (int): Backend OpenCV (mặc định CAP_DSHOW trên Windows).
        width (int | None): Yêu cầu chiều rộng khung hình (None = giữ mặc định).
        height (int | None): Yêu cầu chiều cao khung hình (None = giữ mặc định).
    """

    def __init__(
        self,
        camera_index: int = 0,
        backend: int = cv2.CAP_DSHOW,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.camera_index = camera_index
        self.backend = backend
        self.requested_width = width
        self.requested_height = height
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Context manager: dùng with CameraCapture() as cam:
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Mở / đóng camera
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Mở camera và kiểm tra trạng thái. Raise RuntimeError nếu thất bại."""
        logger.info("Đang mở camera index=%d ...", self.camera_index)
        self._cap = cv2.VideoCapture(self.camera_index, self.backend)

        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Không thể mở camera index={self.camera_index}. "
                "Kiểm tra kết nối thiết bị và index camera."
            )

        # Áp dụng kích thước nếu được yêu cầu
        if self.requested_width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera đã mở: %dx%d @ %.1f FPS", actual_w, actual_h, actual_fps
        )

    def release(self) -> None:
        """Giải phóng camera và đóng mọi cửa sổ OpenCV."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera đã được giải phóng.")
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Đọc frame
    # ------------------------------------------------------------------

    @property
    def is_opened(self) -> bool:
        """True nếu camera đang mở và sẵn sàng đọc."""
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self):
        """
        Đọc một frame từ camera.

        Trả về:
            numpy.ndarray: Frame BGR nếu thành công.

        Raise:
            RuntimeError: Nếu camera chưa mở hoặc không đọc được frame.
        """
        if not self.is_opened:
            raise RuntimeError("Camera chưa được mở. Gọi open() trước.")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError(
                "Không đọc được frame từ camera. "
                "Camera có thể đã bị ngắt kết nối."
            )
        return frame

    # ------------------------------------------------------------------
    # Thông tin camera
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        """Trả về dict thông tin cơ bản của camera đang mở."""
        if not self.is_opened:
            return {"opened": False}
        return {
            "opened": True,
            "index": self.camera_index,
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
        }

