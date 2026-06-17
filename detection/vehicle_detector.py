"""YOLOv8 vehicle detection and counting.

The detector uses pretrained Ultralytics YOLOv8 COCO weights. It never trains
from scratch; the first run downloads the configured pretrained checkpoint if it
is not already present. Inference supports video files, webcam indexes and RTSP
URLs through OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

import config
from core.device import resolve_device
from core.exceptions import ModelLoadError, StreamError
from core.logger import get_logger
from detection.tracker import CentroidTracker

log = get_logger("vehicle_detector")


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    class_id: int

    def as_dict(self) -> dict:
        data = asdict(self)
        data["box"] = list(self.box)
        return data


@dataclass
class FrameAnalysis:
    counts: dict[str, int]
    detections: list[Detection]
    tracks: dict[int, tuple[int, int]]
    frame_index: int
    width: int
    height: int
    total_crossings: int
    roi_density: dict

    def as_dict(self) -> dict:
        return {
            "counts": self.counts,
            "detections": [d.as_dict() for d in self.detections],
            "tracks": {str(k): list(v) for k, v in self.tracks.items()},
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "total_crossings": self.total_crossings,
            "roi_density": self.roi_density,
        }


class VehicleDetector:
    """Pretrained YOLOv8 vehicle detector with lightweight centroid tracking."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> None:
        resolved_model = str(model_path or config.VEHICLE_MODEL)
        # Handle URLs - Ultralytics YOLO() class handles some URLs, 
        # but we ensure it stays as a string.
        if resolved_model.startswith(("http://", "https://")):
            log.info("Using remote YOLO model: %s", resolved_model)
        elif resolved_model.endswith(".pt") and not Path(resolved_model).exists():
            resolved_model = Path(resolved_model).name
        self.model_path = resolved_model
        self.confidence = confidence if confidence is not None else config.VEHICLE_CONF
        self.iou = iou if iou is not None else config.VEHICLE_IOU
        self.imgsz = imgsz or config.INFER_IMGSZ
        self.device = device or resolve_device()
        self.tracker = CentroidTracker()
        self._model = None
        self._model_names: dict[int, str] = {}

    @property
    def model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_path)
                raw_names = getattr(self._model, "names", {}) or {}
                self._model_names = {
                    int(key): str(value) for key, value in raw_names.items()
                }
                log.info("Loaded vehicle YOLO model: %s", self.model_path)
            except Exception as exc:
                raise ModelLoadError(
                    f"Could not load vehicle YOLO model '{self.model_path}': {exc}"
                ) from exc
        return self._model

    def _class_label(self, cls_id: int) -> str | None:
        raw_label = self._model_names.get(cls_id, "").strip().lower()
        if raw_label:
            mapped = config.VEHICLE_LABEL_ALIASES.get(raw_label)
            if mapped:
                return mapped
            return None
        return config.VEHICLE_CLASSES.get(cls_id)

    @staticmethod
    def _scaled_point(point: tuple[int, int], width: int, height: int) -> tuple[int, int]:
        x = int(point[0] * width / max(1, config.ROI_REFERENCE_WIDTH))
        y = int(point[1] * height / max(1, config.ROI_REFERENCE_HEIGHT))
        return x, y

    def _scaled_polygon(
        self, polygon: tuple[tuple[int, int], ...], width: int, height: int
    ) -> np.ndarray:
        return np.array(
            [self._scaled_point(point, width, height) for point in polygon],
            dtype=np.int32,
        )

    def _roi_density(
        self, detections: list[Detection], width: int, height: int
    ) -> dict:
        left_polygon = self._scaled_polygon(config.ROI_LEFT_POLYGON, width, height)
        right_polygon = self._scaled_polygon(config.ROI_RIGHT_POLYGON, width, height)
        threshold_x = int(config.ROI_LANE_THRESHOLD_X * width / max(1, config.ROI_REFERENCE_WIDTH))
        top_y = int(config.ROI_VERTICAL_RANGE[0] * height / max(1, config.ROI_REFERENCE_HEIGHT))
        bottom_y = int(config.ROI_VERTICAL_RANGE[1] * height / max(1, config.ROI_REFERENCE_HEIGHT))

        lanes = {
            "left": {"count": 0, "intensity": "Smooth", "heavy": False},
            "right": {"count": 0, "intensity": "Smooth", "heavy": False},
        }
        for det in detections:
            x1, y1, x2, y2 = det.box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if top_y <= cy <= bottom_y:
                if cx < threshold_x:
                    lanes["left"]["count"] += 1
                else:
                    lanes["right"]["count"] += 1

        for lane in lanes.values():
            lane["heavy"] = lane["count"] > config.ROI_HEAVY_TRAFFIC_THRESHOLD
            lane["intensity"] = "Heavy" if lane["heavy"] else "Smooth"

        total_roi = lanes["left"]["count"] + lanes["right"]["count"]
        return {
            "enabled": True,
            "left_lane": lanes["left"],
            "right_lane": lanes["right"],
            "total_roi_vehicles": total_roi,
            "heavy_threshold": config.ROI_HEAVY_TRAFFIC_THRESHOLD,
            "lane_threshold_x": threshold_x,
            "vertical_range": [top_y, bottom_y],
            "polygons": {
                "left": left_polygon.astype(int).tolist(),
                "right": right_polygon.astype(int).tolist(),
            },
        }

    def set_counting_line(self, y: int) -> None:
        self.tracker.set_counting_line(y)

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> FrameAnalysis:
        """Run YOLOv8 on one BGR frame and return vehicle detections/counts."""
        if frame is None or frame.size == 0:
            raise StreamError("Empty frame received for vehicle detection")

        height, width = frame.shape[:2]
        model = self.model
        results = model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        counts = {name: 0 for name in config.VEHICLE_TYPES}
        boxes_for_tracking: list[tuple[float, float, float, float]] = []

        if results:
            result_names = getattr(results[0], "names", None)
            if result_names:
                self._model_names = {
                    int(key): str(value) for key, value in result_names.items()
                }
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = self._class_label(cls_id)
                    if label is None:
                        continue
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
                    xyxy = (
                        int(max(0, min(width - 1, x1))),
                        int(max(0, min(height - 1, y1))),
                        int(max(0, min(width - 1, x2))),
                        int(max(0, min(height - 1, y2))),
                    )
                    detections.append(
                        Detection(
                            label=label,
                            confidence=round(conf, 4),
                            box=xyxy,
                            class_id=cls_id,
                        )
                    )
                    counts[label] += 1
                    boxes_for_tracking.append((x1, y1, x2, y2))

        tracks = self.tracker.update(boxes_for_tracking)
        roi_density = self._roi_density(detections, width, height)
        return FrameAnalysis(
            counts=counts,
            detections=detections,
            tracks=tracks,
            frame_index=frame_index,
            width=width,
            height=height,
            total_crossings=self.tracker.total_crossings,
            roi_density=roi_density,
        )

    def draw(self, frame: np.ndarray, result: FrameAnalysis) -> np.ndarray:
        """Draw green vehicle boxes, labels and track ids on a copy of frame."""
        output = frame.copy()
        for det in result.detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(output, (x1, y1), (x2, y2), (48, 209, 88), 2)
            text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(
                output,
                text,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (48, 209, 88),
                2,
                cv2.LINE_AA,
            )

        if self.tracker.line_y is not None:
            cv2.line(
                output,
                (0, self.tracker.line_y),
                (result.width, self.tracker.line_y),
                (255, 196, 0),
                2,
            )
        for track_id, (cx, cy) in result.tracks.items():
            cv2.circle(output, (cx, cy), 4, (255, 196, 0), -1)
            cv2.putText(
                output,
                str(track_id),
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 196, 0),
                1,
                cv2.LINE_AA,
            )
        self._draw_roi_overlay(output, result)
        return output

    def _draw_roi_overlay(self, output: np.ndarray, result: FrameAnalysis) -> None:
        roi = result.roi_density or {}
        polygons = roi.get("polygons") or {}
        left = np.array(polygons.get("left", []), dtype=np.int32)
        right = np.array(polygons.get("right", []), dtype=np.int32)
        if left.size:
            cv2.polylines(output, [left], isClosed=True, color=(45, 212, 191), thickness=2)
        if right.size:
            cv2.polylines(output, [right], isClosed=True, color=(56, 189, 248), thickness=2)
        vertical_range = roi.get("vertical_range") or []
        if len(vertical_range) == 2:
            top_y, bottom_y = vertical_range
            overlay = output.copy()
            cv2.rectangle(
                overlay,
                (0, int(top_y)),
                (result.width, int(bottom_y)),
                (20, 184, 166),
                -1,
            )
            cv2.addWeighted(overlay, 0.08, output, 0.92, 0, output)

        left_lane = roi.get("left_lane") or {}
        right_lane = roi.get("right_lane") or {}
        labels = [
            (
                12,
                36,
                f"Left lane: {left_lane.get('count', 0)} | {left_lane.get('intensity', 'Smooth')}",
                (45, 212, 191),
            ),
            (
                max(12, result.width - 420),
                36,
                f"Right lane: {right_lane.get('count', 0)} | {right_lane.get('intensity', 'Smooth')}",
                (56, 189, 248),
            ),
        ]
        for x, y, text, color in labels:
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            cv2.rectangle(output, (x - 8, y - text_h - 10), (x + text_w + 8, y + 10), (2, 6, 23), -1)
            cv2.rectangle(output, (x - 8, y - text_h - 10), (x + text_w + 8, y + 10), color, 1)
            cv2.putText(
                output,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

    def iter_source(
        self,
        source: str | int,
        every_n_frames: int = 1,
        max_frames: int | None = None,
    ) -> Iterator[tuple[np.ndarray, FrameAnalysis]]:
        """Yield annotated frame analyses from file/webcam/RTSP source."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise StreamError(f"Could not open video source: {source}")

        frame_idx = 0
        emitted = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % max(1, every_n_frames) == 0:
                    if self.tracker.line_y is None:
                        self.set_counting_line(frame.shape[0] // 2)
                    result = self.detect(frame, frame_index=frame_idx)
                    yield self.draw(frame, result), result
                    emitted += 1
                    if max_frames is not None and emitted >= max_frames:
                        break
                frame_idx += 1
        finally:
            cap.release()


def normalize_source(source: str | int) -> str | int:
    """Convert webcam-like strings to integer OpenCV indexes."""
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    return text
