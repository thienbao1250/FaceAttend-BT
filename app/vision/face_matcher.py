"""
app/vision/face_matcher.py
So khớp danh tính khuôn mặt (Face Matching) — Giai đoạn 6.

Pipeline:
  embedding (512,) + gallery {employee_id: ndarray(N,512)}
  → MatchResult(status, employee_id, similarity, runner_up_similarity)

Ba trạng thái đầu ra:
  KNOWN     — độ tương đồng đủ cao, danh tính tin cậy
  UNKNOWN   — không khớp bất kỳ nhân viên nào
  UNCERTAIN — gần ngưỡng hoặc hai người cạnh tranh sát nhau

Chiến lược so khớp:
  - Mỗi nhân viên có thể có 1 hoặc nhiều embedding mẫu.
  - Điểm của nhân viên = MAX cosine similarity giữa embedding query và tất cả mẫu.
    Dùng MAX thay vì MEAN để tránh trường hợp một mẫu tốt bị "kéo xuống"
    bởi mẫu góc xấu (phổ biến khi enrollment ít mẫu).
  - Hai ngưỡng độc lập:
      threshold_accept  : sim >= ngưỡng → KNOWN
      threshold_reject  : sim <  ngưỡng → UNKNOWN
      Giữa hai ngưỡng  → UNCERTAIN
  - Nếu hai người có điểm sát nhau (margin_min), kết quả trả về UNCERTAIN
    dù người đứng đầu đã vượt threshold_accept.

Lưu ý bảo mật:
  Gallery (embedding database) là dữ liệu sinh trắc học nhạy cảm.
  Hạn chế quyền truy cập, mã hóa khi lưu, không log embedding thô.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum trạng thái so khớp
# ---------------------------------------------------------------------------

class MatchStatus(Enum):
    """Kết quả định danh một embedding."""
    KNOWN     = "known"      # Đủ tin cậy, có employee_id
    UNKNOWN   = "unknown"    # Không khớp ai trong gallery
    UNCERTAIN = "uncertain"  # Gần ngưỡng hoặc quá sát người khác


# ---------------------------------------------------------------------------
# Kết quả so khớp
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    Kết quả so khớp một embedding với gallery.

    Thuộc tính:
        status              : MatchStatus (KNOWN / UNKNOWN / UNCERTAIN)
        employee_id         : mã nhân viên nếu KNOWN, None nếu không
        similarity          : cosine similarity cao nhất (best match)
        runner_up_id        : mã nhân viên có điểm cao thứ nhì (nếu có)
        runner_up_similarity: điểm cosine của người thứ nhì
        latency_ms          : thời gian so khớp tính bằng millisecond
        num_candidates      : số nhân viên được so sánh
    """
    status:               MatchStatus
    employee_id:          Optional[str]  = None
    similarity:           float          = 0.0
    runner_up_id:         Optional[str]  = None
    runner_up_similarity: float          = 0.0
    latency_ms:           float          = 0.0
    num_candidates:       int            = 0

    def __str__(self) -> str:
        parts = [f"[{self.status.value.upper()}]"]
        if self.employee_id:
            parts.append(f"id={self.employee_id}")
        parts.append(f"sim={self.similarity:.4f}")
        if self.runner_up_id:
            parts.append(f"runner_up={self.runner_up_id}({self.runner_up_similarity:.4f})")
        parts.append(f"n={self.num_candidates}")
        parts.append(f"{self.latency_ms:.1f}ms")
        return " | ".join(parts)

    @property
    def is_known(self) -> bool:
        return self.status == MatchStatus.KNOWN

    @property
    def is_unknown(self) -> bool:
        return self.status == MatchStatus.UNKNOWN

    @property
    def is_uncertain(self) -> bool:
        return self.status == MatchStatus.UNCERTAIN

    @property
    def margin(self) -> float:
        """Khoảng cách giữa người nhất và nhì (0 nếu không có người nhì)."""
        return self.similarity - self.runner_up_similarity


# ---------------------------------------------------------------------------
# Ngưỡng mặc định
# ---------------------------------------------------------------------------

@dataclass
class MatchThresholds:
    """
    Ngưỡng kiểm soát quyết định của FaceMatcher.

    Thuộc tính:
        accept     : sim >= accept → KNOWN          (mặc định 0.35)
        reject     : sim <  reject → UNKNOWN        (mặc định 0.20)
        margin_min : nếu accept nhưng margin < giá trị này → UNCERTAIN
                     Bảo vệ trường hợp hai người cạnh tranh sát nhau.
                     (mặc định 0.07)

    Giải thích vùng:
        sim >= accept  AND  margin >= margin_min  → KNOWN
        sim >= accept  AND  margin <  margin_min  → UNCERTAIN (quá sát)
        reject <= sim < accept                    → UNCERTAIN (gần ngưỡng)
        sim < reject                              → UNKNOWN

    Lưu ý về hiệu chỉnh:
        Các ngưỡng mặc định này chỉ là điểm khởi đầu cho ArcFace w600k_r50.
        Phải hiệu chỉnh lại trên bộ validation riêng của từng môi trường triển khai.
        Điều kiện ánh sáng, góc camera và số lượng mẫu enrollment đều ảnh hưởng.
    """
    accept:     float = 0.35
    reject:     float = 0.20
    margin_min: float = 0.07

    def __post_init__(self) -> None:
        if not (0.0 < self.reject < self.accept <= 1.0):
            raise ValueError(
                f"Phải đảm bảo 0 < reject({self.reject}) < accept({self.accept}) <= 1.0"
            )
        if self.margin_min < 0.0:
            raise ValueError(f"margin_min phải >= 0, nhận {self.margin_min}")


# ---------------------------------------------------------------------------
# Gallery — kho embedding nhân viên
# ---------------------------------------------------------------------------

class FaceGallery:
    """
    Lưu trữ và quản lý embedding của nhân viên đã đăng ký.

    Mỗi nhân viên có thể có nhiều embedding mẫu (N mẫu để tăng độ phủ góc).
    Gallery chỉ làm việc với embedding đã L2-normalize (chuẩn ra từ FaceRecognizer).

    Ví dụ sử dụng::

        gallery = FaceGallery()
        gallery.add("EMP001", emb_array)          # emb_array shape (512,)
        gallery.add("EMP001", another_emb)        # nhiều mẫu cho cùng người
        gallery.add("EMP002", emb_emp2)
        print(gallery)
    """

    def __init__(self) -> None:
        # {employee_id: list of ndarray (512,)}
        self._data: dict[str, list[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, employee_id: str, embedding: np.ndarray) -> None:
        """
        Thêm một embedding mẫu cho nhân viên.

        Tham số:
            employee_id : mã nhân viên duy nhất (string)
            embedding   : ndarray (D,) float32, đã L2-normalize
        """
        self._validate_embedding(embedding)
        if employee_id not in self._data:
            self._data[employee_id] = []
        self._data[employee_id].append(embedding.astype(np.float32))
        logger.debug(
            "Gallery.add: %s → tổng %d mẫu",
            employee_id,
            len(self._data[employee_id]),
        )

    def add_batch(self, employee_id: str, embeddings: np.ndarray) -> None:
        """
        Thêm nhiều embedding mẫu cùng lúc.

        Tham số:
            embeddings : ndarray (N, D) float32
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings phải có shape (N, D), nhận {embeddings.shape}")
        for emb in embeddings:
            self.add(employee_id, emb)

    def remove(self, employee_id: str) -> bool:
        """
        Xóa tất cả mẫu của nhân viên.
        Trả về True nếu tìm thấy và xóa, False nếu không tồn tại.
        """
        if employee_id in self._data:
            del self._data[employee_id]
            logger.info("Gallery.remove: đã xóa %s", employee_id)
            return True
        return False

    def clear(self) -> None:
        """Xóa toàn bộ gallery."""
        self._data.clear()
        logger.info("Gallery.clear: đã xóa tất cả nhân viên.")

    # ------------------------------------------------------------------
    # Lưu / Nạp file (Persistence)
    # ------------------------------------------------------------------

    def save(self, filepath: str | Path) -> None:
        """
        Lưu toàn bộ vector embedding trong gallery ra file nhị phân npz.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = {
            emp_id: np.stack(embs, axis=0)
            for emp_id, embs in self._data.items()
            if embs
        }
        np.savez_compressed(filepath, **data_to_save)
        logger.info("FaceGallery.save: Đã lưu %d nhân viên vào %s", len(data_to_save), filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> FaceGallery:
        """
        Nạp gallery từ file nhị phân npz.
        Nếu file chưa tồn tại, trả về một FaceGallery rỗng.
        """
        filepath = Path(filepath)
        gallery = cls()
        if not filepath.exists():
            logger.info("FaceGallery.load: File %s chưa tồn tại, tạo gallery rỗng.", filepath)
            return gallery

        try:
            with np.load(filepath) as data:
                for emp_id in data.files:
                    matrix = data[emp_id]
                    for emb in matrix:
                        gallery.add(emp_id, emb)
            logger.info(
                "FaceGallery.load: Đã nạp %d nhân viên (%d mẫu) từ %s",
                gallery.size, gallery.total_samples, filepath
            )
        except Exception as exc:
            logger.error("FaceGallery.load lỗi khi đọc %s: %s", filepath, exc)
        return gallery

    # ------------------------------------------------------------------
    # Truy vấn
    # ------------------------------------------------------------------

    @property
    def employee_ids(self) -> list[str]:
        return list(self._data.keys())

    @property
    def size(self) -> int:
        """Số nhân viên trong gallery."""
        return len(self._data)

    @property
    def total_samples(self) -> int:
        """Tổng số embedding mẫu trong gallery."""
        return sum(len(v) for v in self._data.values())

    def get_embeddings(self, employee_id: str) -> list[np.ndarray]:
        """Trả danh sách embedding của nhân viên, hoặc [] nếu không tồn tại."""
        return self._data.get(employee_id, [])

    def has(self, employee_id: str) -> bool:
        return employee_id in self._data

    def get_matrix(self, employee_id: str) -> Optional[np.ndarray]:
        """
        Trả về ma trận (N, D) của tất cả mẫu cho nhân viên.
        Trả None nếu không tồn tại.
        """
        embs = self._data.get(employee_id)
        if not embs:
            return None
        return np.stack(embs, axis=0)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"FaceGallery(employees={self.size}, "
            f"total_samples={self.total_samples})"
        )

    # ------------------------------------------------------------------
    # Nội bộ
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_embedding(emb: np.ndarray) -> None:
        if emb is None or emb.ndim != 1:
            raise ValueError(f"Embedding phải là vector 1-D, nhận {getattr(emb,'shape',None)}")
        if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
            raise ValueError("Embedding chứa NaN hoặc Inf.")


# ---------------------------------------------------------------------------
# FaceMatcher — engine so khớp chính
# ---------------------------------------------------------------------------

class FaceMatcher:
    """
    So khớp một embedding query với tất cả nhân viên trong gallery.

    Tham số:
        gallery    : FaceGallery đã có embedding
        thresholds : MatchThresholds — ngưỡng quyết định (None = dùng mặc định)

    Ví dụ sử dụng::

        matcher = FaceMatcher(gallery)
        result  = matcher.match(query_embedding)

        if result.is_known:
            print(f"Nhân viên: {result.employee_id}  (sim={result.similarity:.3f})")
        elif result.is_uncertain:
            print(f"Không chắc chắn — cần xác minh thêm.")
        else:
            print("Người chưa đăng ký.")
    """

    def __init__(
        self,
        gallery: FaceGallery,
        thresholds: Optional[MatchThresholds] = None,
    ) -> None:
        self.gallery    = gallery
        self.thresholds = thresholds or MatchThresholds()

    # ------------------------------------------------------------------
    # API công khai
    # ------------------------------------------------------------------

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """
        So khớp embedding query với tất cả nhân viên trong gallery.

        Tham số:
            query_embedding : ndarray (D,) float32, đã L2-normalize

        Trả về:
            MatchResult với status, employee_id, similarity, v.v.

        Raise:
            ValueError nếu embedding không hợp lệ hoặc gallery rỗng.
        """
        FaceGallery._validate_embedding(query_embedding)

        if self.gallery.size == 0:
            logger.warning("match() gọi khi gallery rỗng.")
            return MatchResult(
                status=MatchStatus.UNKNOWN,
                similarity=0.0,
                num_candidates=0,
            )

        t0 = time.perf_counter()

        # --- Tính điểm cho từng nhân viên ---
        scores: dict[str, float] = {}
        for emp_id in self.gallery.employee_ids:
            mat = self.gallery.get_matrix(emp_id)   # (N, D)
            if mat is None:
                continue
            # Cosine similarity = dot product (embedding đã L2-normalize)
            sims = mat @ query_embedding             # (N,)
            scores[emp_id] = float(sims.max())      # MAX over samples

        latency_ms = (time.perf_counter() - t0) * 1000

        if not scores:
            return MatchResult(
                status=MatchStatus.UNKNOWN,
                similarity=0.0,
                latency_ms=latency_ms,
                num_candidates=0,
            )

        # --- Sắp xếp theo điểm giảm dần ---
        sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
        best_id    = sorted_ids[0]
        best_sim   = scores[best_id]

        runner_up_id  = sorted_ids[1] if len(sorted_ids) > 1 else None
        runner_up_sim = scores[runner_up_id] if runner_up_id else 0.0

        margin = best_sim - runner_up_sim

        # --- Quyết định ---
        t = self.thresholds
        if best_sim < t.reject:
            status = MatchStatus.UNKNOWN
            matched_id = None
        elif best_sim < t.accept:
            status = MatchStatus.UNCERTAIN
            matched_id = None
        elif margin < t.margin_min:
            # Hai người quá sát nhau → không chắc chắn
            status = MatchStatus.UNCERTAIN
            matched_id = None
        else:
            status = MatchStatus.KNOWN
            matched_id = best_id

        result = MatchResult(
            status=status,
            employee_id=matched_id,
            similarity=best_sim,
            runner_up_id=runner_up_id,
            runner_up_similarity=runner_up_sim,
            latency_ms=latency_ms,
            num_candidates=len(scores),
        )
        logger.debug("match(): %s", result)
        return result

    def match_batch(
        self, query_embeddings: np.ndarray
    ) -> list[MatchResult]:
        """
        So khớp nhiều embedding cùng lúc (xử lý tuần tự).

        Tham số:
            query_embeddings : ndarray (M, D)

        Trả về:
            Danh sách MatchResult, mỗi phần tử ứng với một hàng.
        """
        if query_embeddings.ndim != 2:
            raise ValueError(
                f"query_embeddings phải có shape (M, D), nhận {query_embeddings.shape}"
            )
        return [self.match(query_embeddings[i]) for i in range(len(query_embeddings))]

    # ------------------------------------------------------------------
    # Đánh giá: FAR / FRR
    # ------------------------------------------------------------------

    def evaluate(
        self,
        genuine_pairs: list[tuple[np.ndarray, str]],
        impostor_pairs: list[tuple[np.ndarray, str]],
    ) -> dict:
        """
        Đánh giá FAR và FRR trên tập validation riêng.

        Tham số:
            genuine_pairs  : danh sách (embedding, đúng_employee_id) — người thật
            impostor_pairs : danh sách (embedding, sai_employee_id)   — người giả

        Trả về:
            dict gồm: FAR, FRR, uncertain_rate, total_genuine, total_impostor,
                      và danh sách kết quả chi tiết.

        Lưu ý:
            - genuine: query của người A, expected A.
              False Reject (FRR): hệ thống trả UNKNOWN hoặc UNCERTAIN cho người thật.
            - impostor: query của người B, expected ≠ result.employee_id.
              False Accept (FAR): hệ thống trả KNOWN với sai employee_id.
        """
        genuine_results = [
            (self.match(emb), eid) for emb, eid in genuine_pairs
        ]
        impostor_results = [
            (self.match(emb), expected_wrong_id) for emb, expected_wrong_id in impostor_pairs
        ]

        # Genuine → tính FRR
        g_reject   = sum(1 for r, eid in genuine_results
                         if r.status != MatchStatus.KNOWN or r.employee_id != eid)
        g_uncertain = sum(1 for r, _ in genuine_results if r.is_uncertain)
        frr = g_reject / len(genuine_results) if genuine_results else 0.0

        # Impostor → tính FAR
        # FAR: trả KNOWN với employee_id khác với expected_wrong_id và khác None
        # (tức là nhận nhầm bất kỳ người nào)
        i_accept = sum(
            1 for r, _ in impostor_results
            if r.status == MatchStatus.KNOWN
        )
        i_uncertain = sum(1 for r, _ in impostor_results if r.is_uncertain)
        far = i_accept / len(impostor_results) if impostor_results else 0.0

        uncertain_rate = (g_uncertain + i_uncertain) / (
            len(genuine_results) + len(impostor_results)
        ) if (genuine_results or impostor_results) else 0.0

        result = {
            "FAR":           round(far, 4),
            "FRR":           round(frr, 4),
            "uncertain_rate": round(uncertain_rate, 4),
            "total_genuine":  len(genuine_results),
            "total_impostor": len(impostor_results),
            "genuine_reject": g_reject,
            "impostor_accept": i_accept,
        }
        logger.info("Evaluate: %s", result)
        return result

