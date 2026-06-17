"""LSTM traffic congestion forecasting.

The predictor loads a PyTorch LSTM trained by ``training/train_lstm.py``. If the
model is not available yet, it falls back to a transparent baseline using recent
congestion trend. That keeps the dashboard runnable immediately while making it
clear that proper metrics require running the training pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from core.device import resolve_device
from core.logger import get_logger
from detection.congestion_analyzer import CongestionAnalyzer

log = get_logger("traffic_predictor")


FEATURES = ["total_count", "density", "hour_sin", "hour_cos", "congestion_score"]


@dataclass
class Forecast:
    horizon_min: int
    future_congestion: float
    future_level: str
    source: str

    def as_dict(self) -> dict:
        return {
            "horizon_min": self.horizon_min,
            "future_congestion": self.future_congestion,
            "future_level": self.future_level,
            "source": self.source,
        }


class TrafficLSTM:
    """Small LSTM used by both training and inference."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int):
        import torch

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.1 if num_layers > 1 else 0.0,
                )
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size // 2, output_size),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        self.model = _Model()


class TrafficPredictor:
    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        horizons: list[int] | None = None,
        sequence_length: int | None = None,
    ) -> None:
        self.model_path = Path(model_path or config.LSTM_MODEL_PATH)
        self.scaler_path = Path(scaler_path or config.LSTM_SCALER_PATH)
        self.horizons = horizons or list(config.FORECAST_HORIZONS)
        self.sequence_length = sequence_length or config.SEQUENCE_LENGTH
        self.device = resolve_device()
        self._model = None
        self._scaler: dict | None = None
        self.analyzer = CongestionAnalyzer()

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.exists() or not self.scaler_path.exists():
            return None

        import torch

        checkpoint = torch.load(self.model_path, map_location=self.device)
        input_size = int(checkpoint.get("input_size", len(FEATURES)))
        hidden_size = int(checkpoint.get("hidden_size", config.LSTM_HIDDEN_SIZE))
        num_layers = int(checkpoint.get("num_layers", config.LSTM_NUM_LAYERS))
        output_size = int(checkpoint.get("output_size", len(self.horizons)))
        wrapper = TrafficLSTM(input_size, hidden_size, num_layers, output_size)
        wrapper.model.load_state_dict(checkpoint["state_dict"])
        wrapper.model.to(self.device)
        wrapper.model.eval()
        self._model = wrapper.model
        self._scaler = json.loads(self.scaler_path.read_text(encoding="utf-8"))
        log.info("Loaded LSTM forecaster from %s", self.model_path)
        return self._model

    def _baseline(self, recent_rows: list[dict]) -> list[Forecast]:
        if not recent_rows:
            last = 0.0
            slope = 0.0
        else:
            y = np.array([float(r.get("congestion_score", r.get("density", 0))) for r in recent_rows])
            last = float(y[-1])
            if len(y) >= 3:
                x = np.arange(len(y[-10:]))
                slope = float(np.polyfit(x, y[-10:], 1)[0])
            else:
                slope = 0.0
        forecasts = []
        for horizon in self.horizons:
            intervals = max(1, horizon / max(1, config.FORECAST_INTERVAL_MIN))
            value = float(np.clip(last + slope * intervals, 0, 100))
            forecasts.append(
                Forecast(
                    horizon_min=horizon,
                    future_congestion=round(value, 2),
                    future_level=self.analyzer.classify(value),
                    source="baseline",
                )
            )
        return forecasts

    @staticmethod
    def _row_features(row: dict) -> list[float]:
        ts = str(row.get("timestamp", ""))
        try:
            hour = int(ts[11:13]) + int(ts[14:16]) / 60.0
        except Exception:
            hour = 12.0
        angle = 2.0 * np.pi * hour / 24.0
        return [
            float(row.get("total_count", 0)),
            float(row.get("density", 0)),
            float(np.sin(angle)),
            float(np.cos(angle)),
            float(row.get("congestion_score", row.get("density", 0))),
        ]

    def _scale(self, matrix: np.ndarray) -> np.ndarray:
        if not self._scaler:
            return matrix
        mean = np.array(self._scaler["mean"], dtype=np.float32)
        scale = np.array(self._scaler["scale"], dtype=np.float32)
        return (matrix - mean) / np.where(scale == 0, 1, scale)

    def predict(self, recent_rows: list[dict]) -> list[Forecast]:
        model = self._load_model()
        if model is None or len(recent_rows) < self.sequence_length:
            return self._baseline(recent_rows)

        import torch

        sequence = recent_rows[-self.sequence_length:]
        matrix = np.array([self._row_features(r) for r in sequence], dtype=np.float32)
        matrix = self._scale(matrix)
        x = torch.tensor(matrix[None, :, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            raw = model(x).detach().cpu().numpy()[0]
        forecasts = []
        for horizon, value in zip(self.horizons, raw):
            clipped = float(np.clip(value, 0, 100))
            forecasts.append(
                Forecast(
                    horizon_min=int(horizon),
                    future_congestion=round(clipped, 2),
                    future_level=self.analyzer.classify(clipped),
                    source="lstm",
                )
            )
        return forecasts
