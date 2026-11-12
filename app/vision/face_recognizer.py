"""
app/vision/face_recognizer.py
Trích xuất đặc trưng khuôn mặt (face embedding) bằng ArcFace ONNX.

Pipeline:
  aligned_face_bgr (112×112) → preprocess → ONNX session → L2-normalize → embedding (512,)

Model tương thích:
  - w600k_r50.onnx  (ResNet-50, trained on WebFace600K — InsightFace)
  - w600k_mbf.onnx  (MobileNetV2, nhẹ hơn)

Tham chiếu chuẩn hóa đầu vào (InsightFace/arcface_torch):
  pixel_norm = (pixel - 127.5) / 128.0  →  range ≈ [-1.0, +1.0]
  BGR → RGB trước khi đưa vào model.
  Thứ tự kênh: NCHW.

Đầu ra:
  vector float32 shape (512,) đã L2-normalize.
  - Hai embedding cùng người → cosine similarity ≈ 0.3–0.7+ tùy góc/ánh sáng.
  - Hai embedding khác người → cosine similarity thường < 0.2.

Lưu ý bảo mật:
  Embedding là dữ liệu sinh trắc học nhạy cảm.
  Chỉ truyền/lưu có mã hóa; không ghi vào log thông thường.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hằng số chuẩn hóa ArcFace (InsightFace convention)
# ---------------------------------------------------------------------------

_ARCFACE_MEAN  = 127.5    # trừ mean này khỏi mỗi pixel
_ARCFACE_SCALE = 128.0    # chia cho scale này
_INPUT_SIZE    = 112      # chiều cạnh ảnh đầu vào (112×112)
_EMB_DIM       = 512      # chiều embedding output (ResNet-50 ArcFace)


# ---------------------------------------------------------------------------
# Tiện ích nội bộ
# ---------------------------------------------------------------------------

def _preprocess_aligned(aligned_bgr: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh 112×112 BGR sang tensor NCHW float32 theo chuẩn ArcFace.

    Các bước:
      1. BGR → RGB (đảo kênh màu).
      2. Chuẩn hóa pixel: (x - 127.5) / 128.0 → [-1.0, +1.0].
      3. HWC → NCHW (thêm batch dimension).

    Tham số:
        aligned_bgr : ndarray (112, 112, 3) uint8 BGR

    Trả về:
        ndarray (1, 3, 112, 112) float32
    """
    # Chuyển sang RGB
    rgb = aligned_bgr[:, :, ::-1]  # (H, W, 3) BGR→RGB, không copy thêm bộ nhớ

    # Chuẩn hóa và chuyển sang float32
    normed = (rgb.astype(np.float32) - _ARCFACE_MEAN) / _ARCFACE_SCALE  # (H, W, 3)

    # HWC → CHW → NCHW
    chw   = normed.transpose(2, 0, 1)                   # (3, H, W)
    nchw  = np.expand_dims(chw, axis=0)                 # (1, 3, H, W)
    return nchw


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa L2 vector 1-D.

    Trả về vec / ||vec||₂.
    Nếu norm = 0 (vector không), trả về vector rỗng (tránh NaN).
    """
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        logger.warning("L2 norm ≈ 0 — embedding rỗng hoặc bất thường.")
        return vec
    return vec / norm


# ---------------------------------------------------------------------------
# Lớp chính
# ---------------------------------------------------------------------------

class FaceRecognizer:
    """
    Trích xuất face embedding bằng ArcFace ONNX model.

    Tham số:
        model_path : đường dẫn file .onnx (w600k_r50.onnx hoặc tương đương)
        providers  : ONNX Runtime execution providers
                     Mặc định ["CPUExecutionProvider"].
                     Truyền ["CUDAExecutionProvider", "CPUExecutionProvider"]
                     khi muốn dùng GPU (cần onnxruntime-gpu).

    Thuộc tính công khai:
        embedding_dim : chiều của vector đầu ra (thường 512)
        model_name    : tên file model (để gắn vào metadata embedding)

    Ví dụ sử dụng::

        recognizer = FaceRecognizer("models/w600k_r50.onnx")
        emb = recognizer.get_embedding(aligned_face_bgr)   # (512,) float32, L2-normalized
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: Optional[list[str]] = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model ArcFace: {model_path}\n"
                "Tải về từ: https://github.com/deepinsight/insightface"
            )

        if providers is None:
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3   # ERROR only

        self._session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=providers
        )

        # Xác minh tên input/output
        self._input_name  = self._session.get_inputs()[0].name   # "input.1"
        self._output_name = self._session.get_outputs()[0].name  # "683"

        # Lấy chiều embedding từ model metadata
        output_shape = self._session.get_outputs()[0].shape
        self.embedding_dim: int = int(output_shape[1]) if len(output_shape) > 1 else _EMB_DIM
        self.model_name: str = model_path.name

        # Xác minh provider thực tế được dùng
        actual_providers = self._session.get_providers()
        logger.info(
            "FaceRecognizer đã tải: %s | emb_dim=%d | providers=%s",
            self.model_name,
            self.embedding_dim,
            actual_providers,
        )

    # ------------------------------------------------------------------
    # API công khai
    # ------------------------------------------------------------------

    def get_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """
        Trích xuất và chuẩn hóa embedding từ ảnh khuôn mặt đã căn chỉnh.

        Tham số:
            aligned_face_bgr : ndarray (112, 112, 3) uint8 BGR
                               Phải là ảnh đã qua align_face() từ face_alignment.py.

        Trả về:
            ndarray shape (512,) float32, đã L2-normalize.
            Dùng cosine_similarity() để so sánh với embedding khác.

        Raise:
            ValueError : nếu ảnh không đúng định dạng hoặc kích thước.
            RuntimeError: nếu ONNX session trả về kết quả bất thường.
        """
        self._validate_input(aligned_face_bgr)

        # 1. Tiền xử lý → tensor NCHW
        tensor = _preprocess_aligned(aligned_face_bgr)

        # 2. Chạy model
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: tensor},
        )
        raw_emb = outputs[0]  # (1, 512)

        # 3. Flatten → (512,)
        emb = raw_emb.flatten().astype(np.float32)

        # 4. Kiểm tra tính hợp lệ
        self._validate_output(emb)

        # 5. L2 normalize
        emb_normed = _l2_normalize(emb)

        logger.debug(
            "get_embedding: shape=%s norm_before=%.4f norm_after=%.4f",
            emb.shape,
            float(np.linalg.norm(emb)),
            float(np.linalg.norm(emb_normed)),
        )
        return emb_normed

    def get_embedding_batch(
        self, aligned_faces_bgr: list[np.ndarray]
    ) -> np.ndarray:
        """
        Trích xuất embedding cho nhiều ảnh (xử lý tuần tự, không dùng batch ONNX).

        Tiện lợi khi cần tạo nhiều mẫu trong bước enrollment.

        Tham số:
            aligned_faces_bgr : danh sách ndarray (112, 112, 3) BGR

        Trả về:
            ndarray shape (N, 512) float32 — mỗi hàng là embedding đã normalize.

        Raise:
            ValueError nếu danh sách rỗng.
        """
        if not aligned_faces_bgr:
            raise ValueError("Danh sách ảnh không được rỗng.")

        embeddings = [self.get_embedding(face) for face in aligned_faces_bgr]
        return np.stack(embeddings, axis=0)  # (N, 512)

    # ------------------------------------------------------------------
    # Tiện ích so sánh
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """
        Tính cosine similarity giữa hai embedding đã L2-normalize.

        Nếu embedding đã normalize (||v|| = 1), cosine similarity = dot product.

        Tham số:
            emb_a, emb_b : ndarray (512,) float32, đã L2-normalize

        Trả về:
            float trong khoảng [-1.0, 1.0].
            Giá trị càng gần 1.0 → càng giống nhau.
        """
        return float(np.dot(emb_a, emb_b))

    @staticmethod
    def average_embedding(embeddings: np.ndarray) -> np.ndarray:
        """
        Tính embedding trung bình từ nhiều mẫu (dùng trong enrollment).

        Tham số:
            embeddings : ndarray (N, 512) float32

        Trả về:
            ndarray (512,) float32, đã L2-normalize lại sau khi lấy trung bình.
        """
        mean_emb = embeddings.mean(axis=0)
        return _l2_normalize(mean_emb)

    # ------------------------------------------------------------------
    # Kiểm tra nội bộ
    # ------------------------------------------------------------------

    def _validate_input(self, face_bgr: np.ndarray) -> None:
        """Kiểm tra ảnh đầu vào hợp lệ trước khi chạy model."""
        if face_bgr is None:
            raise ValueError("Ảnh đầu vào không được None.")
        if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
            raise ValueError(
                f"Ảnh phải có shape (H, W, 3), nhận được {face_bgr.shape}."
            )
        h, w = face_bgr.shape[:2]
        if h != _INPUT_SIZE or w != _INPUT_SIZE:
            raise ValueError(
                f"Kích thước ảnh phải là {_INPUT_SIZE}×{_INPUT_SIZE}, "
                f"nhận được {h}×{w}. "
                "Hãy chạy align_face() trước khi gọi get_embedding()."
            )

    @staticmethod
    def _validate_output(emb: np.ndarray) -> None:
        """Kiểm tra embedding đầu ra không chứa NaN/Inf."""
        if np.any(np.isnan(emb)):
            raise RuntimeError("Embedding chứa NaN — model hoặc input có vấn đề.")
        if np.any(np.isinf(emb)):
            raise RuntimeError("Embedding chứa Inf — model hoặc input có vấn đề.")
        if np.all(emb == 0.0):
            raise RuntimeError("Embedding toàn số 0 — input hoặc model có vấn đề.")

