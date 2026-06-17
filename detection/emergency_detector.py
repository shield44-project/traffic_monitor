"""Emergency vehicle detection using fine-tuned YOLOv8 weights.

The module expects a transfer-learned YOLOv8 checkpoint whose classes are:
ambulance, fire_truck and police. The detector intentionally does not fake
emergency detections when the custom checkpoint is missing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

import config
from core.device import resolve_device
from core.exceptions import ModelLoadError, ModelNotFoundError, StreamError, ValidationError
from core.logger import get_logger

log = get_logger("emergency_detector")


@dataclass
class EmergencyDetection:
    vehicle_type: str
    confidence: float
    box: tuple[int, int, int, int]
    class_id: int

    def as_dict(self) -> dict:
        data = asdict(self)
        data["box"] = list(self.box)
        return data


class EmergencyDetector:
    """YOLOv8 emergency vehicle detector."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float | None = None,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path or config.EMERGENCY_MODEL)
        self.confidence = confidence if confidence is not None else config.EMERGENCY_CONF
        self.imgsz = imgsz or config.INFER_IMGSZ
        self.device = device or resolve_device()
        self._model = None
        self._model_names: dict[int, str] = {}

    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise ModelNotFoundError(
                    f"Emergency YOLO weights not found at {self.model_path}. "
                    "Download/install a trained best.pt or run "
                    "training/train_emergency_yolo.py."
                )
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(self.model_path))
                self._model_names = {
                    int(key): str(value)
                    for key, value in (getattr(self._model, "names", {}) or {}).items()
                }
                self._validate_model_labels()
                log.info("Loaded emergency YOLO model: %s", self.model_path)
            except Exception as exc:
                self._model = None
                self._model_names = {}
                raise ModelLoadError(
                    f"Could not load emergency YOLO model '{self.model_path}': {exc}"
                ) from exc
        return self._model

    def _validate_model_labels(self) -> None:
        labels = {name.strip().lower().replace(" ", "_") for name in self._model_names.values()}
        mapped = {
            config.EMERGENCY_LABEL_ALIASES[label]
            for label in labels
            if label in config.EMERGENCY_LABEL_ALIASES
        }
        expected = set(config.EMERGENCY_CLASSES)
        if not mapped & expected:
            raise ModelLoadError(
                "Emergency YOLO checkpoint has labels "
                f"{sorted(labels) or 'unknown'}, but expected at least one of "
                f"{sorted(expected)}. Install a real emergency-vehicle model; "
                "do not use the generic traffic-density checkpoint here."
            )

    def detect(self, frame: np.ndarray) -> list[EmergencyDetection]:
        if frame is None or frame.size == 0:
            raise StreamError("Empty frame received for emergency detection")

        model = self.model

        height, width = frame.shape[:2]
        results = model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections: list[EmergencyDetection] = []
        if results and getattr(results[0], "boxes", None) is not None:
            names = getattr(results[0], "names", None) or self._model_names or {
                i: name for i, name in enumerate(config.EMERGENCY_CLASSES)
            }
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                raw_label = str(names.get(cls_id, "")).strip().lower().replace(" ", "_")
                label = config.EMERGENCY_LABEL_ALIASES.get(raw_label)
                if label not in config.EMERGENCY_CLASSES:
                    continue
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
                detections.append(
                    EmergencyDetection(
                        vehicle_type=label,
                        confidence=round(conf, 4),
                        box=(
                            int(max(0, min(width - 1, x1))),
                            int(max(0, min(height - 1, y1))),
                            int(max(0, min(width - 1, x2))),
                            int(max(0, min(height - 1, y2))),
                        ),
                        class_id=cls_id,
                    )
                )
        return detections

    def draw(self, frame: np.ndarray, detections: list[EmergencyDetection]) -> np.ndarray:
        """Draw red emergency boxes and a top alert banner."""
        output = frame.copy()
        if detections:
            cv2.rectangle(output, (0, 0), (output.shape[1], 42), (0, 0, 190), -1)
            cv2.putText(
                output,
                f"EMERGENCY VEHICLE ALERT ({len(detections)})",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"{det.vehicle_type} {det.confidence:.2f}"
            cv2.putText(
                output,
                label,
                (x1, max(58 if detections else 18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return output


def validate_emergency_model_file(path: str | Path) -> dict:
    """Load a YOLO checkpoint and verify emergency-vehicle labels."""
    try:
        from ultralytics import YOLO

        model = YOLO(str(path))
        names = {int(key): str(value) for key, value in (getattr(model, "names", {}) or {}).items()}
    except Exception as exc:
        raise ValidationError(f"Could not inspect emergency model '{path}': {exc}") from exc

    labels = {name.strip().lower().replace(" ", "_") for name in names.values()}
    mapped = {
        config.EMERGENCY_LABEL_ALIASES[label]
        for label in labels
        if label in config.EMERGENCY_LABEL_ALIASES
    }
    expected = set(config.EMERGENCY_CLASSES)
    if not mapped & expected:
        raise ValidationError(
            "Emergency model labels do not match ambulance/fire_truck/police. "
            f"Found: {', '.join(sorted(labels)) or 'unknown'}."
        )
    return {"names": names, "matched_labels": sorted(mapped & expected)}
