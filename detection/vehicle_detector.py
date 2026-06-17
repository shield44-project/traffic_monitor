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

    def as_dict(self) -> dict:
        return {
            "counts": self.counts,
            "detections": [d.as_dict() for d in self.detections],
            "tracks": {str(k): list(v) for k, v in self.tracks.items()},
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "total_crossings": self.total_crossings,
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

    @property
    def model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_path)
                log.info("Loaded vehicle YOLO model: %s", self.model_path)
            except Exception as exc:
                raise ModelLoadError(
                    f"Could not load vehicle YOLO model '{self.model_path}': {exc}"
                ) from exc
        return self._model

    def set_counting_line(self, y: int) -> None:
        self.tracker.set_counting_line(y)

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> FrameAnalysis:
        """Run YOLOv8 on one BGR frame and return vehicle detections/counts."""
        if frame is None or frame.size == 0:
            raise StreamError("Empty frame received for vehicle detection")

        height, width = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=list(config.VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        detections: list[Detection] = []
        counts = {name: 0 for name in config.VEHICLE_CLASSES.values()}
        boxes_for_tracking: list[tuple[float, float, float, float]] = []

        if results:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = config.VEHICLE_CLASSES.get(cls_id)
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
        return FrameAnalysis(
            counts=counts,
            detections=detections,
            tracks=tracks,
            frame_index=frame_index,
            width=width,
            height=height,
            total_crossings=self.tracker.total_crossings,
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
        return output

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
