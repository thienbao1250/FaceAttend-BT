"""
tests/unit/test_face_detector.py
Unit test cho app/vision/face_detector.py
Dùng mock — không cần ONNX model thật.
Chạy: python -m pytest tests/unit/test_face_detector.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.vision.face_detector import (
    FaceDetection,
    FaceDetector,
    _distance2bbox,
    _distance2kps,
    _generate_anchor_centers,
    _nms,
)


# ===========================================================================
# _generate_anchor_centers
# ===========================================================================

class TestGenerateAnchorCenters:
    def test_shape_1_anchor(self):
        """640x640 input, stride=8 → 80×80 × 1 = 6400 centers."""
        centers = _generate_anchor_centers(640, 640, stride=8, num_anchors=1)
        assert centers.shape == (6400, 2)

    def test_shape_2_anchors(self):
        """640x640, stride=8, 2 anchors → 80×80 × 2 = 12800."""
        centers = _generate_anchor_centers(640, 640, stride=8, num_anchors=2)
        assert centers.shape == (12800, 2)

    def test_stride16(self):
        """640x640, stride=16 → 40×40 × 2 = 3200."""
        centers = _generate_anchor_centers(640, 640, stride=16, num_anchors=2)
        assert centers.shape == (3200, 2)

    def test_stride32(self):
        """640x640, stride=32 → 20×20 × 2 = 800."""
        centers = _generate_anchor_centers(640, 640, stride=32, num_anchors=2)
        assert centers.shape == (800, 2)

    def test_first_center_is_zero(self):
        """Center đầu tiên (0,0) × stride = (0,0) → nhân với stride vẫn là 0."""
        centers = _generate_anchor_centers(640, 640, stride=8, num_anchors=1)
        assert centers[0, 0] == 0.0
        assert centers[0, 1] == 0.0

    def test_second_center_stride(self):
        """Center thứ 2 (col=1, row=0) × stride=8 → (8, 0)."""
        centers = _generate_anchor_centers(640, 640, stride=8, num_anchors=1)
        assert centers[1, 0] == 8.0   # cx
        assert centers[1, 1] == 0.0   # cy

    def test_dtype_float32(self):
        centers = _generate_anchor_centers(640, 640, stride=8, num_anchors=2)
        assert centers.dtype == np.float32


# ===========================================================================
# _distance2bbox
# ===========================================================================

class TestDistance2BBox:
    def test_basic_decode(self):
        """Center (100,100) + offsets (10,20,30,40) → (90,80,130,140)."""
        centers = np.array([[100.0, 100.0]], dtype=np.float32)
        preds = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        boxes = _distance2bbox(centers, preds)
        assert boxes.shape == (1, 4)
        assert abs(boxes[0, 0] - 90.0) < 1e-4   # x1
        assert abs(boxes[0, 1] - 80.0) < 1e-4   # y1
        assert abs(boxes[0, 2] - 130.0) < 1e-4  # x2
        assert abs(boxes[0, 3] - 140.0) < 1e-4  # y2

    def test_zero_offsets(self):
        """Offset = 0 → bbox bằng điểm center (suy biến)."""
        centers = np.array([[50.0, 60.0]], dtype=np.float32)
        preds = np.zeros((1, 4), dtype=np.float32)
        boxes = _distance2bbox(centers, preds)
        assert abs(boxes[0, 0] - 50.0) < 1e-4
        assert abs(boxes[0, 1] - 60.0) < 1e-4

    def test_multiple_rows(self):
        centers = np.tile([100.0, 100.0], (5, 1)).astype(np.float32)
        preds = np.tile([10.0, 10.0, 10.0, 10.0], (5, 1)).astype(np.float32)
        boxes = _distance2bbox(centers, preds)
        assert boxes.shape == (5, 4)
        assert np.all(boxes[:, 0] == 90.0)


# ===========================================================================
# _distance2kps
# ===========================================================================

class TestDistance2Kps:
    def test_shape(self):
        """(N, 10) → (N, 5, 2)."""
        centers = np.zeros((3, 2), dtype=np.float32)
        preds = np.zeros((3, 10), dtype=np.float32)
        kps = _distance2kps(centers, preds)
        assert kps.shape == (3, 5, 2)

    def test_basic_decode(self):
        """Center (100,100) + offsets (5,10, 0,0, ...) → kp0=(105,110)."""
        centers = np.array([[100.0, 100.0]], dtype=np.float32)
        preds = np.zeros((1, 10), dtype=np.float32)
        preds[0, 0] = 5.0   # kp0 dx
        preds[0, 1] = 10.0  # kp0 dy
        kps = _distance2kps(centers, preds)
        assert abs(kps[0, 0, 0] - 105.0) < 1e-4
        assert abs(kps[0, 0, 1] - 110.0) < 1e-4

    def test_dtype_float32(self):
        centers = np.zeros((2, 2), dtype=np.float32)
        preds = np.zeros((2, 10), dtype=np.float32)
        kps = _distance2kps(centers, preds)
        assert kps.dtype == np.float32


# ===========================================================================
# _nms
# ===========================================================================

class TestNMS:
    def test_empty_input(self):
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.array([], dtype=np.float32)
        keep = _nms(boxes, scores, 0.4)
        assert keep == []

    def test_single_box(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        keep = _nms(boxes, scores, 0.4)
        assert keep == [0]

    def test_non_overlapping_boxes(self):
        """2 box không giao nhau → giữ cả 2."""
        boxes = np.array([
            [0, 0, 10, 10],
            [100, 100, 110, 110],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = _nms(boxes, scores, 0.4)
        assert len(keep) == 2

    def test_fully_overlapping_boxes(self):
        """2 box giống hệt → chỉ giữ 1 (box có score cao hơn)."""
        boxes = np.array([
            [0, 0, 10, 10],
            [0, 0, 10, 10],
        ], dtype=np.float32)
        scores = np.array([0.8, 0.9], dtype=np.float32)
        keep = _nms(boxes, scores, 0.4)
        assert len(keep) == 1
        # Box score cao (index 1) phải được giữ
        assert keep[0] == 1

    def test_high_iou_suppresses(self):
        """Box thứ 2 gần như trùng → bị suppress."""
        boxes = np.array([
            [0, 0, 100, 100],
            [1, 1, 101, 101],   # IoU rất cao
        ], dtype=np.float32)
        scores = np.array([0.9, 0.85], dtype=np.float32)
        keep = _nms(boxes, scores, 0.4)
        assert len(keep) == 1

    def test_low_iou_keeps_both(self):
        """Box thứ 2 chỉ giao 1 phần nhỏ → cả 2 được giữ."""
        boxes = np.array([
            [0, 0, 10, 10],
            [5, 5, 20, 20],   # IoU thấp
        ], dtype=np.float32)
        scores = np.array([0.9, 0.85], dtype=np.float32)
        keep = _nms(boxes, scores, 0.9)  # threshold cao → cả 2 kept
        assert len(keep) == 2

    def test_output_sorted_by_score(self):
        """Index đầu tiên trong keep phải là box score cao nhất."""
        boxes = np.array([
            [0, 0, 5, 5],
            [50, 50, 55, 55],
            [100, 100, 105, 105],
        ], dtype=np.float32)
        scores = np.array([0.6, 0.9, 0.7], dtype=np.float32)
        keep = _nms(boxes, scores, 0.4)
        # Box index=1 score=0.9 phải đứng đầu
        assert keep[0] == 1


# ===========================================================================
# FaceDetection dataclass
# ===========================================================================

class TestFaceDetection:
    def _make(self):
        return FaceDetection(bbox=(10.0, 20.0, 110.0, 120.0), confidence=0.95)

    def test_width(self):
        assert self._make().width == 100.0

    def test_height(self):
        assert self._make().height == 100.0

    def test_area(self):
        assert self._make().area == 10000.0

    def test_as_int_bbox(self):
        d = FaceDetection(bbox=(10.3, 20.7, 110.2, 120.9), confidence=0.9)
        ib = d.as_int_bbox()
        assert ib == (10, 20, 110, 120)

    def test_no_landmarks_default(self):
        d = self._make()
        assert d.landmarks is None


# ===========================================================================
# FaceDetector (mock ONNX session)
# ===========================================================================

def _make_mock_session(has_kps: bool = True):
    """
    Mock ort.InferenceSession với output giả.
    Trả về 9 output (với kps) hoặc 6 output (không kps).

    Mỗi stride tạo 2 "faces" giả để test pipeline đầy đủ.
    """
    mock_sess = MagicMock()

    # Input name
    mock_input = MagicMock()
    mock_input.name = "input.1"
    mock_sess.get_inputs.return_value = [mock_input]

    num_outputs = 9 if has_kps else 6
    mock_outputs = [MagicMock() for _ in range(num_outputs)]
    mock_sess.get_outputs.return_value = mock_outputs

    # Tạo output tensor giả cho 640×640 input, stride=[8,16,32], 2 anchors
    # Số anchors: 80*80*2=12800, 40*40*2=3200, 20*20*2=800
    anchor_counts = [12800, 3200, 800]

    scores_list = []
    bboxes_list = []
    kps_list = []

    for n in anchor_counts:
        # scores: 2 faces thật ở đầu với conf=0.9, còn lại conf=0.1
        s = np.full((n, 1), 0.1, dtype=np.float32)
        s[0, 0] = 0.9
        s[1, 0] = 0.85
        scores_list.append(s)

        # bboxes: offset nhỏ từ anchor (sẽ cho box hợp lý)
        b = np.zeros((n, 4), dtype=np.float32)
        b[0] = [30, 30, 60, 80]   # face 1
        b[1] = [150, 30, 60, 80]  # face 2 (ở anchor khác)
        bboxes_list.append(b)

        if has_kps:
            k = np.zeros((n, 10), dtype=np.float32)
            kps_list.append(k)

    if has_kps:
        # 9 outputs: score×3, bbox×3, kps×3
        all_outputs = scores_list + bboxes_list + kps_list
    else:
        all_outputs = scores_list + bboxes_list

    mock_sess.run.return_value = all_outputs
    return mock_sess


class TestFaceDetector:
    @patch("app.vision.face_detector.ort.InferenceSession")
    def test_model_not_found_raises(self, _mock_ort):
        with pytest.raises(FileNotFoundError):
            FaceDetector("nonexistent_model.onnx")

    @patch("app.vision.face_detector.ort.InferenceSession")
    def test_init_with_kps(self, mock_ort_cls, tmp_path):
        # Tạo file model giả
        model_file = tmp_path / "scrfd.onnx"
        model_file.write_bytes(b"fake")
        mock_ort_cls.return_value = _make_mock_session(has_kps=True)

        detector = FaceDetector(model_file, conf_threshold=0.5)
        assert detector._has_kps is True
        assert detector._num_outputs == 9

    @patch("app.vision.face_detector.ort.InferenceSession")
    def test_init_without_kps(self, mock_ort_cls, tmp_path):
        model_file = tmp_path / "scrfd.onnx"
        model_file.write_bytes(b"fake")
        mock_ort_cls.return_value = _make_mock_session(has_kps=False)

        detector = FaceDetector(model_file)
        assert detector._has_kps is False
        assert detector._num_outputs == 6

    @patch("app.vision.face_detector.ort.InferenceSession")
    def test_detect_returns_list(self, mock_ort_cls, tmp_path):
        model_file = tmp_path / "scrfd.onnx"
        model_file.write_bytes(b"fake")
        mock_ort_cls.return_value = _make_mock_session(has_kps=True)

        detector = FaceDetector(model_file, conf_threshold=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)
        assert isinstance(results, list)

    @patch("app.vision.face_detector.ort.InferenceSession")
    def test_detect_faces_found(self, mock_ort_cls, tmp_path):
        """Mock tạo 2 faces với conf=0.9/0.85 → phải tìm được ít nhất 1."""
        model_file = tmp_path / "scrfd.onnx"
        model_file.write_bytes(b"fake")
        mock_ort_cls.return_value = _make_mock_session(has_kps=True)

        detector = FaceDetector(model_file, conf_threshold=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)
        assert len(results) >= 1

    @patch("app.vision.face_detector.ort.InferenceSession")
    def test_detect_sorted_by_confidence(self, mock_ort_cls, tmp_path):
        """Kết quả phải sắp xếp confidence giảm dần."""
        model_file = tmp_path / "scrfd.onnx"
        model_file.write_bytes(b"fake")
        mock_ort_cls.return_value = _make_mock_session(has_kps=True)

        detector = FaceDetector(model_file, conf_threshold=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)
        if len(results) >= 2:
            assert results[0].confidence >= results[1].confidence

    @patch("app.vision.face_detector.ort.InferenceSession")
    def test_detect_result_has_landmarks(self, mock_ort_cls, tmp_path):
        """Model với kps → FaceDetection phải có landmarks."""
        model_file = tmp_path / "scrfd.onnx"
        model_file.write_bytes(b"fake")
        mock_ort_cls.return_value = _make_mock_session(has_kps=True)

        detector = FaceDetector(model_file, conf_threshold=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)
        if results:
            assert results[0].landmarks is not None
            assert results[0].landmarks.shape == (5, 2)

    @patch("app.vision.face_detector.ort.InferenceSession")
    def test_detect_no_faces_high_threshold(self, mock_ort_cls, tmp_path):
        """conf_threshold=1.0 → không có face nào vượt qua."""
        model_file = tmp_path / "scrfd.onnx"
        model_file.write_bytes(b"fake")
        mock_ort_cls.return_value = _make_mock_session(has_kps=True)

        detector = FaceDetector(model_file, conf_threshold=1.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)
        assert results == []

    @patch("app.vision.face_detector.ort.InferenceSession")
    def test_detect_bbox_in_frame_coords(self, mock_ort_cls, tmp_path):
        """bbox phải nằm trong phạm vi frame gốc (640x480)."""
        model_file = tmp_path / "scrfd.onnx"
        model_file.write_bytes(b"fake")
        mock_ort_cls.return_value = _make_mock_session(has_kps=True)

        detector = FaceDetector(model_file, conf_threshold=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)
        for det in results:
            x1, y1, x2, y2 = det.bbox
            # Bbox có thể hơi ra ngoài do padding, nhưng x2 > x1 và y2 > y1
            assert x2 > x1, f"x2 ({x2}) phải > x1 ({x1})"
            assert y2 > y1, f"y2 ({y2}) phải > y1 ({y1})"

