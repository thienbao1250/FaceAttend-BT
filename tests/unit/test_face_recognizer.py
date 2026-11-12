"""
tests/unit/test_face_recognizer.py
Unit test cho app/vision/face_recognizer.py (Giai đoạn 5 — Face Embedding)

Chạy: python -m pytest tests/unit/test_face_recognizer.py -v

Lưu ý: Các test dùng synthetic data (không cần model thật) để kiểm tra
  - _preprocess_aligned: định dạng tensor đầu ra
  - _l2_normalize: chuẩn hóa L2 đúng
  - cosine_similarity: tính toán chính xác
  - average_embedding: trung bình và normalize
  - _validate_input: bắt đúng lỗi đầu vào sai
  - _validate_output: bắt NaN/Inf/zero
  - FaceRecognizer với mock session: pipeline end-to-end không cần ONNX thật
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Thêm root vào sys.path để import app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.vision.face_recognizer import (
    _ARCFACE_MEAN,
    _ARCFACE_SCALE,
    _INPUT_SIZE,
    FaceRecognizer,
    _l2_normalize,
    _preprocess_aligned,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_aligned(h: int = 112, w: int = 112, seed: int = 42) -> np.ndarray:
    """Tạo ảnh aligned giả (H, W, 3) uint8."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _unit_vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    """Trả về vector ngẫu nhiên đã L2-normalize."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_mock_recognizer(emb_output: np.ndarray) -> FaceRecognizer:
    """
    Tạo FaceRecognizer với ONNX session giả trả embedding cho trước.

    emb_output : (1, 512) float32
    """
    with patch("app.vision.face_recognizer.ort.InferenceSession") as mock_cls:
        mock_session = MagicMock()

        # Mô phỏng inputs/outputs metadata
        mock_input = MagicMock()
        mock_input.name = "input.1"
        mock_session.get_inputs.return_value = [mock_input]

        mock_output = MagicMock()
        mock_output.name = "683"
        mock_output.shape = [1, 512]
        mock_session.get_outputs.return_value = [mock_output]

        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session.run.return_value = [emb_output]

        mock_cls.return_value = mock_session

        # Tạo file giả để qua FileNotFoundError
        fake_path = Path("/fake/w600k_r50.onnx")
        with patch("app.vision.face_recognizer.Path.exists", return_value=True):
            rec = FaceRecognizer(fake_path)

        return rec


# ===========================================================================
# Tests _preprocess_aligned
# ===========================================================================

class TestPreprocessAligned:
    def test_output_shape(self):
        """Tensor phải là (1, 3, 112, 112)."""
        img = _make_aligned()
        t = _preprocess_aligned(img)
        assert t.shape == (1, 3, _INPUT_SIZE, _INPUT_SIZE)

    def test_output_dtype(self):
        """Tensor phải là float32."""
        img = _make_aligned()
        t = _preprocess_aligned(img)
        assert t.dtype == np.float32

    def test_normalization_range(self):
        """Giá trị pixel sau normalize phải nằm trong [-1.0, +1.0] ± epsilon."""
        img = np.zeros((112, 112, 3), dtype=np.uint8)   # min pixel = 0
        t_min = _preprocess_aligned(img)
        expected_min = (0.0 - _ARCFACE_MEAN) / _ARCFACE_SCALE  # = -0.99609375
        assert abs(t_min.min() - expected_min) < 1e-5

        img_max = np.full((112, 112, 3), 255, dtype=np.uint8)  # max pixel = 255
        t_max = _preprocess_aligned(img_max)
        expected_max = (255.0 - _ARCFACE_MEAN) / _ARCFACE_SCALE  # ≈ 0.99609375
        assert abs(t_max.max() - expected_max) < 1e-5

    def test_bgr_to_rgb_channel_swap(self):
        """
        Kiểm tra BGR→RGB swap đúng hướng.

        Ảnh input: BGR → channel 0 (B=200), channel 1 (G=0), channel 2 (R=0).
        Sau swap:  RGB → channel 0 (R=0),   channel 1 (G=0),  channel 2 (B=200).

        Vậy trong tensor NCHW:
          tensor[0, 0, ...] = R channel = pixel gốc channel 2 (R=0)
          tensor[0, 2, ...] = B channel = pixel gốc channel 0 (B=200)
        """
        img = np.zeros((112, 112, 3), dtype=np.uint8)
        img[:, :, 0] = 200   # B channel (index 0 trong BGR) = 200; G=R=0

        t = _preprocess_aligned(img)

        # R channel (index 0 trong tensor) lấy từ R của ảnh gốc (index 2 trong BGR) = 0
        expected_r_chan = (0.0   - _ARCFACE_MEAN) / _ARCFACE_SCALE
        # B channel (index 2 trong tensor) lấy từ B của ảnh gốc (index 0 trong BGR) = 200
        expected_b_chan = (200.0 - _ARCFACE_MEAN) / _ARCFACE_SCALE

        assert abs(t[0, 0, 0, 0] - expected_r_chan) < 1e-5   # tensor[:, R_chan, ...]
        assert abs(t[0, 2, 0, 0] - expected_b_chan) < 1e-5   # tensor[:, B_chan, ...]

    def test_deterministic(self):
        """Cùng input → cùng output."""
        img = _make_aligned(seed=7)
        assert np.array_equal(_preprocess_aligned(img), _preprocess_aligned(img))


# ===========================================================================
# Tests _l2_normalize
# ===========================================================================

class TestL2Normalize:
    def test_norm_is_one(self):
        """Vector sau normalize phải có norm = 1.0 ± epsilon."""
        v = np.random.default_rng(0).standard_normal(512).astype(np.float32)
        n = _l2_normalize(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6

    def test_direction_preserved(self):
        """Hướng vector không đổi sau normalize."""
        v = np.array([3.0, 4.0], dtype=np.float32)
        n = _l2_normalize(v)
        expected = np.array([0.6, 0.8], dtype=np.float32)
        np.testing.assert_allclose(n, expected, atol=1e-6)

    def test_zero_vector_no_nan(self):
        """Vector rỗng không được sinh NaN."""
        v = np.zeros(512, dtype=np.float32)
        n = _l2_normalize(v)
        assert not np.any(np.isnan(n))

    def test_single_element(self):
        v = np.array([5.0], dtype=np.float32)
        n = _l2_normalize(v)
        assert abs(n[0] - 1.0) < 1e-6


# ===========================================================================
# Tests cosine_similarity
# ===========================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Hai vector giống hệt nhau → similarity = 1.0."""
        v = _unit_vec()
        sim = FaceRecognizer.cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_opposite_vectors(self):
        """Hai vector đối nhau → similarity = -1.0."""
        v = _unit_vec()
        sim = FaceRecognizer.cosine_similarity(v, -v)
        assert abs(sim - (-1.0)) < 1e-6

    def test_orthogonal_vectors(self):
        """Hai vector trực giao → similarity ≈ 0."""
        v1 = np.zeros(512, dtype=np.float32); v1[0] = 1.0
        v2 = np.zeros(512, dtype=np.float32); v2[1] = 1.0
        sim = FaceRecognizer.cosine_similarity(v1, v2)
        assert abs(sim) < 1e-6

    def test_range(self):
        """Similarity nằm trong [-1, 1] với vector ngẫu nhiên."""
        v1 = _unit_vec(seed=1)
        v2 = _unit_vec(seed=2)
        sim = FaceRecognizer.cosine_similarity(v1, v2)
        assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6

    def test_symmetry(self):
        """cosine_similarity(a, b) == cosine_similarity(b, a)."""
        v1 = _unit_vec(seed=3)
        v2 = _unit_vec(seed=4)
        assert abs(
            FaceRecognizer.cosine_similarity(v1, v2)
            - FaceRecognizer.cosine_similarity(v2, v1)
        ) < 1e-6


# ===========================================================================
# Tests average_embedding
# ===========================================================================

class TestAverageEmbedding:
    def test_output_shape(self):
        embs = np.stack([_unit_vec(seed=i) for i in range(5)])
        avg = FaceRecognizer.average_embedding(embs)
        assert avg.shape == (512,)

    def test_output_normalized(self):
        """Embedding trung bình phải được L2-normalize."""
        embs = np.stack([_unit_vec(seed=i) for i in range(5)])
        avg = FaceRecognizer.average_embedding(embs)
        assert abs(np.linalg.norm(avg) - 1.0) < 1e-6

    def test_single_sample(self):
        """Trung bình của 1 mẫu == chính mẫu đó (đã normalize)."""
        v = _unit_vec(seed=99)
        embs = v[np.newaxis, :]  # (1, 512)
        avg = FaceRecognizer.average_embedding(embs)
        np.testing.assert_allclose(avg, v, atol=1e-6)

    def test_identical_samples(self):
        """Trung bình của N mẫu giống hệt == mẫu đó."""
        v = _unit_vec(seed=5)
        embs = np.tile(v, (10, 1))
        avg = FaceRecognizer.average_embedding(embs)
        np.testing.assert_allclose(avg, v, atol=1e-6)


# ===========================================================================
# Tests _validate_input
# ===========================================================================

class TestValidateInput:
    """Dùng mock recognizer để test _validate_input."""

    @pytest.fixture(autouse=True)
    def setup(self):
        raw_emb = np.ones((1, 512), dtype=np.float32)
        self.rec = _make_mock_recognizer(raw_emb)

    def test_valid_input_no_raise(self):
        img = _make_aligned()
        # Không raise
        self.rec._validate_input(img)

    def test_none_raises(self):
        with pytest.raises(ValueError, match="None"):
            self.rec._validate_input(None)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="shape"):
            self.rec._validate_input(np.zeros((112, 112)))

    def test_wrong_channels_raises(self):
        with pytest.raises(ValueError, match="shape"):
            self.rec._validate_input(np.zeros((112, 112, 1), dtype=np.uint8))

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError, match="112×112"):
            self.rec._validate_input(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_wrong_height_raises(self):
        with pytest.raises(ValueError, match="112×112"):
            self.rec._validate_input(np.zeros((100, 112, 3), dtype=np.uint8))


# ===========================================================================
# Tests _validate_output
# ===========================================================================

class TestValidateOutput:
    def test_valid_output_no_raise(self):
        emb = np.ones(512, dtype=np.float32)
        FaceRecognizer._validate_output(emb)  # không raise

    def test_nan_raises(self):
        emb = np.ones(512, dtype=np.float32)
        emb[10] = float("nan")
        with pytest.raises(RuntimeError, match="NaN"):
            FaceRecognizer._validate_output(emb)

    def test_inf_raises(self):
        emb = np.ones(512, dtype=np.float32)
        emb[0] = float("inf")
        with pytest.raises(RuntimeError, match="Inf"):
            FaceRecognizer._validate_output(emb)

    def test_all_zero_raises(self):
        emb = np.zeros(512, dtype=np.float32)
        with pytest.raises(RuntimeError, match="toàn số 0"):
            FaceRecognizer._validate_output(emb)


# ===========================================================================
# Tests get_embedding (pipeline qua mock)
# ===========================================================================

class TestGetEmbedding:
    def test_output_shape(self):
        """get_embedding phải trả (512,)."""
        raw = np.random.default_rng(0).standard_normal((1, 512)).astype(np.float32)
        rec = _make_mock_recognizer(raw)
        emb = rec.get_embedding(_make_aligned())
        assert emb.shape == (512,)

    def test_output_dtype(self):
        raw = np.ones((1, 512), dtype=np.float32)
        rec = _make_mock_recognizer(raw)
        emb = rec.get_embedding(_make_aligned())
        assert emb.dtype == np.float32

    def test_output_normalized(self):
        """Embedding phải có norm ≈ 1.0."""
        raw = np.random.default_rng(1).standard_normal((1, 512)).astype(np.float32)
        rec = _make_mock_recognizer(raw)
        emb = rec.get_embedding(_make_aligned())
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_no_nan_in_output(self):
        raw = np.random.default_rng(2).standard_normal((1, 512)).astype(np.float32)
        rec = _make_mock_recognizer(raw)
        emb = rec.get_embedding(_make_aligned())
        assert not np.any(np.isnan(emb))

    def test_deterministic(self):
        """Cùng ảnh → cùng embedding."""
        raw = np.random.default_rng(3).standard_normal((1, 512)).astype(np.float32)
        rec = _make_mock_recognizer(raw)
        img = _make_aligned(seed=11)
        emb1 = rec.get_embedding(img)
        emb2 = rec.get_embedding(img)
        np.testing.assert_array_equal(emb1, emb2)

    def test_raises_on_bad_input_shape(self):
        raw = np.ones((1, 512), dtype=np.float32)
        rec = _make_mock_recognizer(raw)
        with pytest.raises(ValueError):
            rec.get_embedding(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_raises_on_nan_output(self):
        """Nếu model trả NaN, get_embedding phải raise RuntimeError."""
        raw = np.full((1, 512), float("nan"), dtype=np.float32)
        rec = _make_mock_recognizer(raw)
        with pytest.raises(RuntimeError, match="NaN"):
            rec.get_embedding(_make_aligned())


# ===========================================================================
# Tests get_embedding_batch
# ===========================================================================

class TestGetEmbeddingBatch:
    def test_output_shape(self):
        raw = np.random.default_rng(10).standard_normal((1, 512)).astype(np.float32)
        rec = _make_mock_recognizer(raw)
        faces = [_make_aligned(seed=i) for i in range(4)]
        batch = rec.get_embedding_batch(faces)
        assert batch.shape == (4, 512)

    def test_each_row_normalized(self):
        raw = np.random.default_rng(11).standard_normal((1, 512)).astype(np.float32)
        rec = _make_mock_recognizer(raw)
        faces = [_make_aligned(seed=i) for i in range(3)]
        batch = rec.get_embedding_batch(faces)
        for i in range(3):
            assert abs(np.linalg.norm(batch[i]) - 1.0) < 1e-5

    def test_empty_list_raises(self):
        raw = np.ones((1, 512), dtype=np.float32)
        rec = _make_mock_recognizer(raw)
        with pytest.raises(ValueError, match="rỗng"):
            rec.get_embedding_batch([])


# ===========================================================================
# Tests embedding_dim và model_name
# ===========================================================================

class TestModelMetadata:
    def test_embedding_dim(self):
        raw = np.ones((1, 512), dtype=np.float32)
        rec = _make_mock_recognizer(raw)
        assert rec.embedding_dim == 512

    def test_model_name(self):
        raw = np.ones((1, 512), dtype=np.float32)
        rec = _make_mock_recognizer(raw)
        # Tên file từ path "/fake/w600k_r50.onnx"
        assert rec.model_name == "w600k_r50.onnx"


# ===========================================================================
# Tests ngữ nghĩa embedding (semantic sanity check)
# ===========================================================================

class TestEmbeddingSimilaritySanity:
    """
    Kiểm tra tính nhất quán chiều hướng của embedding qua mock.

    Vì dùng mock, chỉ kiểm tra:
      - same input → same output (deterministic)
      - two distinct random raw embeddings → distinct outputs
    Test nhận diện thực trên model thật được thực hiện trong integration test.
    """

    def test_same_face_same_embedding(self):
        """Mock trả cùng raw → hai lần call phải cho cùng embedding."""
        raw = np.random.default_rng(99).standard_normal((1, 512)).astype(np.float32)
        rec = _make_mock_recognizer(raw)
        img = _make_aligned(seed=0)
        e1 = rec.get_embedding(img)
        e2 = rec.get_embedding(img)
        sim = FaceRecognizer.cosine_similarity(e1, e2)
        assert abs(sim - 1.0) < 1e-5, "Cùng input, cùng mock → similarity phải = 1.0"

    def test_different_raw_embeddings_distinct(self):
        """
        Hai raw embedding khác nhau → similarity < 1.0.
        Mô phỏng hai người có đặc trưng khác nhau.
        """
        raw_a = np.random.default_rng(10).standard_normal((1, 512)).astype(np.float32)
        raw_b = np.random.default_rng(20).standard_normal((1, 512)).astype(np.float32)

        rec_a = _make_mock_recognizer(raw_a)
        rec_b = _make_mock_recognizer(raw_b)

        emb_a = rec_a.get_embedding(_make_aligned(seed=0))
        emb_b = rec_b.get_embedding(_make_aligned(seed=0))

        sim = FaceRecognizer.cosine_similarity(emb_a, emb_b)
        assert sim < 0.99, f"Hai raw khác nhau phải cho similarity < 0.99, nhận {sim:.4f}"
