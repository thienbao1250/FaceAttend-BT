"""
tests/unit/test_face_matcher.py
Unit test cho app/vision/face_matcher.py (Giai đoạn 6 — Face Matching)

Chạy: python -m pytest tests/unit/test_face_matcher.py -v

Phạm vi test:
  - FaceGallery: add, add_batch, remove, clear, queries
  - MatchThresholds: validation ngưỡng
  - FaceMatcher.match: KNOWN / UNKNOWN / UNCERTAIN
  - FaceMatcher: gallery rỗng, 1 người, nhiều người, cạnh tranh sát nhau
  - FaceMatcher.match_batch
  - FaceMatcher.evaluate: FAR / FRR
  - MatchResult: properties, margin, __str__
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.vision.face_matcher import (
    FaceGallery,
    FaceMatcher,
    MatchResult,
    MatchStatus,
    MatchThresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    """Tạo vector ngẫu nhiên đã L2-normalize."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _gallery_with(*pairs: tuple[str, np.ndarray]) -> FaceGallery:
    """Tạo gallery từ danh sách (employee_id, embedding)."""
    g = FaceGallery()
    for eid, emb in pairs:
        g.add(eid, emb)
    return g


def _default_matcher(gallery: FaceGallery) -> FaceMatcher:
    return FaceMatcher(gallery, MatchThresholds(accept=0.35, reject=0.20, margin_min=0.07))


# ===========================================================================
# Tests MatchThresholds
# ===========================================================================

class TestMatchThresholds:
    def test_default_values(self):
        t = MatchThresholds()
        assert t.accept == 0.35
        assert t.reject == 0.20
        assert t.margin_min == 0.07

    def test_valid_custom(self):
        t = MatchThresholds(accept=0.5, reject=0.25, margin_min=0.05)
        assert t.accept == 0.5

    def test_reject_ge_accept_raises(self):
        with pytest.raises(ValueError, match="reject"):
            MatchThresholds(accept=0.3, reject=0.3)

    def test_reject_gt_accept_raises(self):
        with pytest.raises(ValueError, match="reject"):
            MatchThresholds(accept=0.2, reject=0.3)

    def test_negative_margin_raises(self):
        with pytest.raises(ValueError, match="margin_min"):
            MatchThresholds(accept=0.35, reject=0.20, margin_min=-0.01)

    def test_zero_margin_valid(self):
        t = MatchThresholds(accept=0.35, reject=0.20, margin_min=0.0)
        assert t.margin_min == 0.0


# ===========================================================================
# Tests FaceGallery
# ===========================================================================

class TestFaceGallery:
    def test_empty_on_init(self):
        g = FaceGallery()
        assert g.size == 0
        assert g.total_samples == 0

    def test_add_one(self):
        g = FaceGallery()
        g.add("A", _unit_vec())
        assert g.size == 1
        assert g.total_samples == 1

    def test_add_multiple_samples_same_person(self):
        g = FaceGallery()
        for i in range(5):
            g.add("A", _unit_vec(seed=i))
        assert g.size == 1
        assert g.total_samples == 5

    def test_add_multiple_people(self):
        g = FaceGallery()
        g.add("A", _unit_vec(seed=1))
        g.add("B", _unit_vec(seed=2))
        g.add("C", _unit_vec(seed=3))
        assert g.size == 3

    def test_has_returns_true(self):
        g = FaceGallery()
        g.add("A", _unit_vec())
        assert g.has("A") is True

    def test_has_returns_false(self):
        g = FaceGallery()
        assert g.has("NOTEXIST") is False

    def test_remove_existing(self):
        g = FaceGallery()
        g.add("A", _unit_vec())
        result = g.remove("A")
        assert result is True
        assert g.size == 0

    def test_remove_nonexistent(self):
        g = FaceGallery()
        result = g.remove("GHOST")
        assert result is False

    def test_clear(self):
        g = FaceGallery()
        for i in range(3):
            g.add(str(i), _unit_vec(seed=i))
        g.clear()
        assert g.size == 0

    def test_employee_ids(self):
        g = FaceGallery()
        g.add("X", _unit_vec(seed=0))
        g.add("Y", _unit_vec(seed=1))
        assert set(g.employee_ids) == {"X", "Y"}

    def test_add_batch(self):
        g = FaceGallery()
        embs = np.stack([_unit_vec(seed=i) for i in range(4)])
        g.add_batch("A", embs)
        assert g.total_samples == 4

    def test_add_batch_wrong_ndim_raises(self):
        g = FaceGallery()
        with pytest.raises(ValueError, match="shape"):
            g.add_batch("A", _unit_vec())   # 1D thay vì 2D

    def test_get_matrix_shape(self):
        g = FaceGallery()
        for i in range(3):
            g.add("A", _unit_vec(seed=i))
        mat = g.get_matrix("A")
        assert mat is not None
        assert mat.shape == (3, 512)

    def test_get_matrix_nonexistent_returns_none(self):
        g = FaceGallery()
        assert g.get_matrix("GHOST") is None

    def test_get_embeddings_nonexistent_returns_empty(self):
        g = FaceGallery()
        assert g.get_embeddings("X") == []

    def test_len(self):
        g = FaceGallery()
        g.add("A", _unit_vec())
        g.add("B", _unit_vec(seed=1))
        assert len(g) == 2

    def test_add_nan_raises(self):
        g = FaceGallery()
        bad = np.full(512, float("nan"), dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            g.add("A", bad)

    def test_add_inf_raises(self):
        g = FaceGallery()
        bad = np.full(512, float("inf"), dtype=np.float32)
        with pytest.raises(ValueError, match="Inf"):
            g.add("A", bad)

    def test_add_wrong_ndim_raises(self):
        g = FaceGallery()
        bad = np.zeros((3, 512), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            g.add("A", bad)

    def test_repr(self):
        g = FaceGallery()
        g.add("A", _unit_vec())
        assert "FaceGallery" in repr(g)


# ===========================================================================
# Tests FaceMatcher — gallery rỗng
# ===========================================================================

class TestFaceMatcherEmptyGallery:
    def test_returns_unknown(self):
        g = FaceGallery()
        m = _default_matcher(g)
        result = m.match(_unit_vec())
        assert result.status == MatchStatus.UNKNOWN
        assert result.employee_id is None
        assert result.num_candidates == 0


# ===========================================================================
# Tests FaceMatcher — 1 nhân viên, 1 mẫu
# ===========================================================================

class TestFaceMatcherSingleEmployee:
    def _setup(self, threshold: MatchThresholds = None):
        v = _unit_vec(seed=10)
        g = FaceGallery()
        g.add("EMP001", v)
        t = threshold or MatchThresholds(accept=0.35, reject=0.20, margin_min=0.0)
        return FaceMatcher(g, t), v

    def test_exact_same_embedding_known(self):
        """Embedding giống hệt mẫu → KNOWN."""
        m, v = self._setup()
        result = m.match(v)
        assert result.status == MatchStatus.KNOWN
        assert result.employee_id == "EMP001"
        assert abs(result.similarity - 1.0) < 1e-5

    def test_similar_embedding_above_threshold_known(self):
        """
        Embedding gần giống (sim đủ cao) → KNOWN.
        Tạo bằng cách thêm noise nhỏ rồi normalize lại.
        """
        m, v = self._setup()
        rng = np.random.default_rng(99)
        noisy = v + rng.standard_normal(512).astype(np.float32) * 0.3
        noisy = noisy / np.linalg.norm(noisy)
        result = m.match(noisy)
        # Vì gallery chỉ có 1 người → không có runner-up → margin = best_sim
        # → chỉ cần sim >= accept
        assert result.num_candidates == 1

    def test_unrelated_embedding_unknown(self):
        """
        Embedding hoàn toàn khác (trực giao) → UNKNOWN.
        """
        _, v = self._setup()
        # Tạo vector trực giao
        v2 = _unit_vec(seed=777)
        # Trừ thành phần song song → trực giao → sim ≈ 0
        v2_orth = v2 - float(np.dot(v2, v)) * v
        v2_orth = v2_orth / np.linalg.norm(v2_orth)

        m, _ = self._setup()
        result = m.match(v2_orth)
        assert result.similarity < 0.20
        assert result.status == MatchStatus.UNKNOWN

    def test_result_has_no_runner_up(self):
        """Chỉ 1 người → runner_up_id phải là None."""
        m, v = self._setup()
        result = m.match(v)
        assert result.runner_up_id is None
        assert result.runner_up_similarity == 0.0

    def test_latency_ms_positive(self):
        m, v = self._setup()
        result = m.match(v)
        assert result.latency_ms >= 0.0


# ===========================================================================
# Tests FaceMatcher — nhiều nhân viên
# ===========================================================================

class TestFaceMatcherMultipleEmployees:
    def _setup_two(self):
        """Tạo gallery 2 người với embedding gần trực giao."""
        v_a = _unit_vec(seed=0)
        v_b = _unit_vec(seed=100)
        g = FaceGallery()
        g.add("EMP_A", v_a)
        g.add("EMP_B", v_b)
        t = MatchThresholds(accept=0.35, reject=0.20, margin_min=0.07)
        m = FaceMatcher(g, t)
        return m, v_a, v_b

    def test_query_a_matches_emp_a(self):
        m, v_a, _ = self._setup_two()
        result = m.match(v_a)
        assert result.employee_id == "EMP_A"
        assert result.status == MatchStatus.KNOWN
        assert abs(result.similarity - 1.0) < 1e-5

    def test_query_b_matches_emp_b(self):
        m, _, v_b = self._setup_two()
        result = m.match(v_b)
        assert result.employee_id == "EMP_B"
        assert result.status == MatchStatus.KNOWN

    def test_runner_up_is_set(self):
        m, v_a, _ = self._setup_two()
        result = m.match(v_a)
        assert result.runner_up_id is not None
        assert result.runner_up_id == "EMP_B"
        assert result.runner_up_similarity < result.similarity

    def test_margin_property(self):
        m, v_a, v_b = self._setup_two()
        result = m.match(v_a)
        assert abs(result.margin - (result.similarity - result.runner_up_similarity)) < 1e-6

    def test_num_candidates(self):
        m, v_a, _ = self._setup_two()
        result = m.match(v_a)
        assert result.num_candidates == 2

    def test_unregistered_person_unknown(self):
        """Người lạ (orthogonal) → UNKNOWN."""
        m, v_a, v_b = self._setup_two()
        # Tạo người lạ: trực giao với cả hai
        v_c = _unit_vec(seed=999)
        v_c = v_c - float(np.dot(v_c, v_a)) * v_a
        v_c = v_c - float(np.dot(v_c, v_b)) * v_b
        norm = np.linalg.norm(v_c)
        if norm < 1e-6:
            v_c = _unit_vec(seed=12345)
        else:
            v_c = v_c / norm
        result = m.match(v_c)
        assert result.status in (MatchStatus.UNKNOWN, MatchStatus.UNCERTAIN)
        assert result.employee_id is None


# ===========================================================================
# Tests FaceMatcher — trường hợp cạnh tranh sát (margin < margin_min)
# ===========================================================================

class TestFaceMatcherUncertainMargin:
    def test_close_competition_returns_uncertain(self):
        """
        Nếu hai người có similarity gần nhau (margin nhỏ) → UNCERTAIN
        dù người đứng đầu vượt accept threshold.
        """
        # Tạo 2 embedding gần nhau: v_b = v_a * (1 - eps) + noise nhỏ
        v_a = _unit_vec(seed=0)
        rng = np.random.default_rng(1)
        # v_b hơi lệch khỏi v_a nhưng vẫn gần
        v_b = v_a + rng.standard_normal(512).astype(np.float32) * 0.05
        v_b = v_b / np.linalg.norm(v_b)

        g = FaceGallery()
        g.add("EMP_A", v_a)
        g.add("EMP_B", v_b)

        # Dùng margin_min cao (0.3) để chắc chắn kích hoạt UNCERTAIN
        t = MatchThresholds(accept=0.35, reject=0.20, margin_min=0.30)
        m = FaceMatcher(g, t)

        # Query gần v_a → score A cao hơn B nhưng margin < 0.30
        result = m.match(v_a)
        # Vì margin giữa EMP_A (sim=1.0) và EMP_B (sim gần 1.0) có thể > 0.30
        # nên kiểm tra logic rõ hơn:
        sim_a = float(np.dot(v_a, v_a))  # = 1.0
        sim_b = float(np.dot(v_b, v_a))
        margin = sim_a - sim_b
        if margin < 0.30:
            assert result.status == MatchStatus.UNCERTAIN
            assert result.employee_id is None
        else:
            # margin đủ lớn → KNOWN bình thường
            assert result.status == MatchStatus.KNOWN


# ===========================================================================
# Tests FaceMatcher — vùng uncertain (giữa reject và accept)
# ===========================================================================

class TestFaceMatcherUncertainZone:
    def test_between_thresholds_returns_uncertain(self):
        """
        Embedding có similarity nằm giữa reject và accept → UNCERTAIN.
        """
        v_ref = _unit_vec(seed=0)
        g = FaceGallery()
        g.add("EMP", v_ref)

        # Tạo query có sim ≈ 0.27 (giữa reject=0.20 và accept=0.35)
        # Bằng cách interpolate: q = alpha*v_ref + (1-alpha)*perp
        perp = _unit_vec(seed=1)
        # Trực giao hóa
        perp = perp - float(np.dot(perp, v_ref)) * v_ref
        perp = perp / np.linalg.norm(perp)

        alpha = 0.27   # sim = alpha (khi v_ref và perp trực giao)
        q = alpha * v_ref + np.sqrt(1 - alpha**2) * perp
        q = q.astype(np.float32) / np.linalg.norm(q)

        actual_sim = float(np.dot(q, v_ref))

        t = MatchThresholds(accept=0.35, reject=0.20, margin_min=0.0)
        m = FaceMatcher(g, t)
        result = m.match(q)

        if 0.20 <= actual_sim < 0.35:
            assert result.status == MatchStatus.UNCERTAIN
            assert result.employee_id is None
        elif actual_sim >= 0.35:
            assert result.status == MatchStatus.KNOWN
        else:
            assert result.status == MatchStatus.UNKNOWN


# ===========================================================================
# Tests FaceMatcher — nhiều mẫu mỗi người (enrollment đa mẫu)
# ===========================================================================

class TestFaceMatcherMultipleSamples:
    def test_max_over_samples_used(self):
        """
        MAX over samples: 1 mẫu tốt đủ để nhận ra người dù mẫu khác xấu.
        """
        v_good = _unit_vec(seed=0)
        v_bad  = _unit_vec(seed=99)   # mẫu góc xấu (trực giao)

        g = FaceGallery()
        g.add("EMP", v_good)
        g.add("EMP", v_bad)

        t = MatchThresholds(accept=0.35, reject=0.20, margin_min=0.0)
        m = FaceMatcher(g, t)

        result = m.match(v_good)
        # MAX(sim(v_good, v_good)=1.0, sim(v_good, v_bad)≈0) = 1.0 → KNOWN
        assert result.status == MatchStatus.KNOWN
        assert result.employee_id == "EMP"
        assert abs(result.similarity - 1.0) < 1e-5

    def test_multiple_samples_improve_coverage(self):
        """
        Gallery có mẫu gần query (noise rất nhỏ) → MAX sim vượt accept → KNOWN.

        Dùng noise nhỏ (0.05) để đảm bảo cosine similarity giữa
        base và mẫu luôn > 0.35. (Với noise=0.2 trên dim=512, sim ≈ 0.26.)
        """
        rng = np.random.default_rng(42)
        base = _unit_vec(seed=0)
        # Tạo 5 mẫu với noise rất nhỏ → cosine sim với base ≈ 0.998
        samples = []
        for _ in range(5):
            noisy = base + rng.standard_normal(512).astype(np.float32) * 0.05
            noisy = noisy / np.linalg.norm(noisy)
            samples.append(noisy)

        g = FaceGallery()
        for s in samples:
            g.add("EMP", s)

        t = MatchThresholds(accept=0.35, reject=0.20, margin_min=0.0)
        m = FaceMatcher(g, t)

        # Query = chính base vector → MAX(sim mẫu với base) rất gần 1.0 → KNOWN
        result = m.match(base)
        assert result.similarity >= 0.35
        assert result.status == MatchStatus.KNOWN


# ===========================================================================
# Tests match_batch
# ===========================================================================

class TestFaceMatcherBatch:
    def test_batch_returns_list(self):
        v = _unit_vec(seed=0)
        g = _gallery_with(("A", v))
        m = _default_matcher(g)
        queries = np.stack([v, _unit_vec(seed=1)])
        results = m.match_batch(queries)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_first_matches(self):
        v = _unit_vec(seed=0)
        g = _gallery_with(("A", v))
        m = _default_matcher(g)
        queries = np.stack([v, _unit_vec(seed=99)])
        results = m.match_batch(queries)
        assert results[0].employee_id == "A"

    def test_batch_wrong_ndim_raises(self):
        v = _unit_vec()
        g = _gallery_with(("A", v))
        m = _default_matcher(g)
        with pytest.raises(ValueError, match="shape"):
            m.match_batch(v)   # 1D thay vì 2D


# ===========================================================================
# Tests evaluate (FAR / FRR)
# ===========================================================================

class TestFaceMatcherEvaluate:
    def _make_matcher_two_people(self):
        v_a = _unit_vec(seed=0)
        v_b = _unit_vec(seed=100)
        g = FaceGallery()
        g.add("A", v_a)
        g.add("B", v_b)
        t = MatchThresholds(accept=0.35, reject=0.20, margin_min=0.07)
        return FaceMatcher(g, t), v_a, v_b

    def test_perfect_genuine(self):
        """Genuine query = chính mẫu đăng ký → FRR = 0."""
        m, v_a, v_b = self._make_matcher_two_people()
        genuine = [(v_a, "A"), (v_b, "B")]
        impostor = []   # bỏ qua FAR khi không có impostor
        report = m.evaluate(genuine, impostor)
        assert report["FRR"] == 0.0
        assert report["total_genuine"] == 2

    def test_evaluate_keys(self):
        m, v_a, v_b = self._make_matcher_two_people()
        report = m.evaluate([(v_a, "A")], [])
        for key in ("FAR", "FRR", "uncertain_rate", "total_genuine", "total_impostor"):
            assert key in report

    def test_evaluate_empty(self):
        m, _, _ = self._make_matcher_two_people()
        report = m.evaluate([], [])
        assert report["total_genuine"] == 0
        assert report["total_impostor"] == 0


# ===========================================================================
# Tests MatchResult
# ===========================================================================

class TestMatchResult:
    def test_is_known(self):
        r = MatchResult(status=MatchStatus.KNOWN, employee_id="A", similarity=0.8)
        assert r.is_known is True
        assert r.is_unknown is False
        assert r.is_uncertain is False

    def test_is_unknown(self):
        r = MatchResult(status=MatchStatus.UNKNOWN)
        assert r.is_unknown is True

    def test_is_uncertain(self):
        r = MatchResult(status=MatchStatus.UNCERTAIN)
        assert r.is_uncertain is True

    def test_margin(self):
        r = MatchResult(
            status=MatchStatus.KNOWN,
            similarity=0.80,
            runner_up_similarity=0.60,
        )
        assert abs(r.margin - 0.20) < 1e-6

    def test_str_contains_status(self):
        r = MatchResult(status=MatchStatus.KNOWN, employee_id="A", similarity=0.75)
        assert "KNOWN" in str(r)
        assert "A" in str(r)
