"""
app/vision/face_detector.py
Phát hiện khuôn mặt bằng SCRFD (ONNX Runtime).

Pipeline:
  frame BGR → preprocess_for_detector() → ONNX session → decode → NMS → FaceDetection[]

Chỉ trả danh sách FaceDetection — không gán tên, không ghi chấm công.

Tương thích với các model scrfd_*g_bnkps.onnx của insightface:
  - scrfd_500m_bnkps.onnx  (nhẹ nhất, CPU)
  - scrfd_2.5g_bnkps.onnx  (cân bằng tốt)
  - scrfd_10g_bnkps.onnx   (chính xác nhất, cần GPU)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.vision.preprocessing import LetterboxInfo, preprocess_for_detector

logger = logging.getLogger(__name__)

# SCRFD mặc định: 3 strides, 2 anchors mỗi vị trí
_DEFAULT_STRIDES = [8, 16, 32]
_NUM_ANCHORS = 2


# ---------------------------------------------------------------------------
# Dataclass kết quả phát hiện một khuôn mặt
# ---------------------------------------------------------------------------

@dataclass
class FaceDetection:
    """
    Kết quả phát hiện một khuôn mặt trong frame gốc.

    Thuộc tính:
        bbox        : (x1, y1, x2, y2) pixel, tọa độ trong frame gốc (BGR từ camera)
        confidence  : điểm tin cậy [0.0, 1.0]
        landmarks   : ndarray (5, 2) toạ độ 5 điểm mốc (mắt trái, mắt phải, mũi,
                      miệng trái, miệng phải) trong frame gốc. None nếu model không
                      hỗ trợ.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    landmarks: np.ndarray | None = field(default=None)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_int_bbox(self) -> tuple[int, int, int, int]:
        """Trả bbox dạng int (x1, y1, x2, y2) để dùng với OpenCV."""
        return tuple(int(v) for v in self.bbox)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Hàm giải mã nội bộ
# ---------------------------------------------------------------------------

def _generate_anchor_centers(
    input_h: int, input_w: int, stride: int, num_anchors: int
) -> np.ndarray:
    """
    Tạo tọa độ trung tâm anchor trong không gian ảnh đầu vào (pixel).

    Trả về:
        ndarray shape (feat_h * feat_w * num_anchors, 2) — (cx, cy)
    """
    feat_h = input_h // stride
    feat_w = input_w // stride
    # mgrid trả (row, col) → đảo thành (col/x, row/y)
    cy_grid, cx_grid = np.mgrid[:feat_h, :feat_w]
    centers = np.stack([cx_grid, cy_grid], axis=-1).astype(np.float32) * stride
    centers = centers.reshape(-1, 2)
    if num_anchors > 1:
        centers = np.stack([centers] * num_anchors, axis=1).reshape(-1, 2)
    return centers


def _distance2bbox(anchor_centers: np.ndarray, bbox_preds: np.ndarray) -> np.ndarray:
    """
    Giải mã bbox từ anchor center + distance offsets [left, top, right, bottom].

    bbox_preds shape: (N, 4)  — đơn vị pixel trong ảnh input (đã scale bởi stride)
    Trả về: (N, 4) dạng [x1, y1, x2, y2]
    """
    x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
    y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
    x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
    y2 = anchor_centers[:, 1] + bbox_preds[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(anchor_centers: np.ndarray, kps_preds: np.ndarray) -> np.ndarray:
    """
    Giải mã 5 landmarks từ anchor center + xy-offsets.

    kps_preds shape: (N, 10)  — x0,y0, x1,y1, ..., x4,y4
    Trả về: (N, 5, 2)
    """
    preds = []
    for i in range(0, kps_preds.shape[1], 2):
        px = anchor_centers[:, 0] + kps_preds[:, i]
        py = anchor_centers[:, 1] + kps_preds[:, i + 1]
        preds.append(np.stack([px, py], axis=-1))
    return np.stack(preds, axis=1)   # (N, 5, 2)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """
    Non-Maximum Suppression đơn giản bằng numpy.

    Tham số:
        boxes         : (N, 4) [x1, y1, x2, y2]
        scores        : (N,)
        iou_threshold : ngưỡng IoU để loại box trùng

    Trả về:
        Danh sách index các box giữ lại.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0, ix2 - ix1 + 1)
        inter_h = np.maximum(0, iy2 - iy1 + 1)
        inter = inter_w * inter_h

        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]

    return keep


# ---------------------------------------------------------------------------
# Lớp chính
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    Phát hiện khuôn mặt bằng SCRFD ONNX model.

    Tham số:
        model_path      : đường dẫn tới file .onnx
        conf_threshold  : chỉ giữ detection có confidence >= ngưỡng này (mặc định 0.5)
        nms_threshold   : ngưỡng IoU cho NMS (mặc định 0.4)
        input_size      : (width, height) đầu vào model (mặc định 640x640)
        providers       : ONNX Runtime execution providers
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
        min_face_size: int = 60,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model SCRFD: {model_path}\n"
                "Chạy scripts/download_models.py để tải về."
            )

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size  # (W, H)
        self.min_face_size = min_face_size  # cạnh tối thiểu (px) trên frame gốc

        if providers is None:
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3   # ERROR only — tắt info/warning của ORT
        self._session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._num_outputs = len(self._session.get_outputs())
        self._has_kps = self._num_outputs == 9  # 9 = với landmarks; 6 = không có

        logger.info(
            "FaceDetector đã tải: %s | outputs=%d | kps=%s | conf=%.2f | nms=%.2f | min_face=%dpx",
            model_path.name,
            self._num_outputs,
            self._has_kps,
            conf_threshold,
            nms_threshold,
            min_face_size,
        )

    # ------------------------------------------------------------------
    # Phát hiện khuôn mặt trong một frame
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        """
        Phát hiện tất cả khuôn mặt trong frame BGR.

        Tham số:
            frame_bgr : ndarray (H, W, 3) uint8, BGR từ OpenCV

        Trả về:
            Danh sách FaceDetection theo thứ tự confidence giảm dần.
            Danh sách rỗng nếu không phát hiện khuôn mặt nào.
        """
        # 1. Tiền xử lý
        tensor, lb_info = preprocess_for_detector(frame_bgr, self.input_size)

        # 2. Chạy model
        outputs = self._session.run(None, {self._input_name: tensor})

        # 3. Giải mã từng stride
        input_w, input_h = self.input_size
        fmc = len(_DEFAULT_STRIDES)   # feature map count per output type (3)

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_kps: list[np.ndarray] = []

        for i, stride in enumerate(_DEFAULT_STRIDES):
            # Outputs được xếp theo: score_s8, score_s16, score_s32,
            #                        bbox_s8,  bbox_s16,  bbox_s32,
            #                        kps_s8,   kps_s16,   kps_s32  (nếu có)
            scores = outputs[i].ravel()              # (N,)
            bbox_preds = outputs[i + fmc]            # (N, 4)
            kps_preds = outputs[i + fmc * 2] if self._has_kps else None

            pos_idx = np.where(scores >= self.conf_threshold)[0]
            if pos_idx.size == 0:
                continue

            # Lọc theo conf threshold
            pos_scores = scores[pos_idx]
            pos_bboxes = bbox_preds[pos_idx] * stride

            # Tạo anchor centers cho stride này
            centers = _generate_anchor_centers(input_h, input_w, stride, _NUM_ANCHORS)
            pos_centers = centers[pos_idx]

            # Giải mã bbox (tọa độ trong không gian tensor letterbox)
            decoded_boxes = _distance2bbox(pos_centers, pos_bboxes)
            all_boxes.append(decoded_boxes)
            all_scores.append(pos_scores)

            if self._has_kps and kps_preds is not None:
                pos_kps = kps_preds[pos_idx] * stride
                decoded_kps = _distance2kps(pos_centers, pos_kps)  # (N, 5, 2)
                all_kps.append(decoded_kps)

        if not all_boxes:
            return []

        # 4. Gộp tất cả strides
        boxes_all = np.concatenate(all_boxes, axis=0)   # (M, 4)
        scores_all = np.concatenate(all_scores, axis=0)  # (M,)
        kps_all = np.concatenate(all_kps, axis=0) if all_kps else None

        # 5. NMS
        keep = _nms(boxes_all, scores_all, self.nms_threshold)

        # 6. Ánh xạ tọa độ về frame gốc
        boxes_kept = boxes_all[keep]
        scores_kept = scores_all[keep]
        kps_kept = kps_all[keep] if kps_all is not None else None

        boxes_orig = lb_info.scale_boxes_to_original(boxes_kept)

        kps_orig = None
        if kps_kept is not None:
            # scale_landmarks_to_original nhận (N, K, 2)
            kps_orig = lb_info.scale_landmarks_to_original(kps_kept)

        # 7. Tạo kết quả — lọc bbox không hợp lệ và quá nhỏ
        results: list[FaceDetection] = []
        for j in range(len(keep)):
            x1, y1, x2, y2 = boxes_orig[j]
            box_w = x2 - x1
            box_h = y2 - y1

            # Bỏ qua bbox âm, đảo chiều, hoặc nhỏ hơn min_face_size
            # (thường là ghost detection từ stride s=8 ở vùng vai/cổ)
            if box_w < self.min_face_size or box_h < self.min_face_size:
                logger.debug(
                    "Bỏ bbox quá nhỏ: %.0fx%.0f (min=%dpx)",
                    box_w, box_h, self.min_face_size,
                )
                continue

            lm = kps_orig[j] if kps_orig is not None else None
            results.append(
                FaceDetection(
                    bbox=tuple(boxes_orig[j].tolist()),
                    confidence=float(scores_kept[j]),
                    landmarks=lm,
                )
            )

        # Sắp xếp theo confidence giảm dần
        results.sort(key=lambda d: d.confidence, reverse=True)
        logger.debug("Phát hiện %d khuôn mặt (sau min_size filter).", len(results))
        return results

    # ------------------------------------------------------------------
    # Tiện ích lọc kết quả
    # ------------------------------------------------------------------

    @staticmethod
    def largest_face(
        detections: list["FaceDetection"],
        min_confidence: float = 0.0,
    ) -> "FaceDetection | None":
        """
        Trả về khuôn mặt có diện tích lớn nhất (gần camera nhất).

        Thường dùng cho app điểm danh 1 người: loại bỏ ghost detection
        và khuôn mặt nhỏ ở phía sau.

        Tham số:
            detections     : danh sách FaceDetection từ detect()
            min_confidence : chỉ xét các detection >= ngưỡng này (mặc định 0.0
                             — tức dùng ngưỡng đã lọc sẵn trong detect())

        Trả về:
            FaceDetection lớn nhất, hoặc None nếu danh sách rỗng / không đủ ngưỡng.
        """
        candidates = [
            d for d in detections if d.confidence >= min_confidence
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.area)

