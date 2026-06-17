"""Vehicle emission prediction with XGBoost regressors and factor fallback.

The trained models estimate pollutant totals from traffic features. Before
training, the predictor uses a documented factor-based fallback using
vehicle-class g/km factors anchored to EPA/MOVES/EEA-style road inventory
values. This gives vehicle-specific contribution estimates immediately.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from core.logger import get_logger
from prediction.emission_factors import (
    EMISSION_FACTORS_G_PER_KM,
    GWP_100,
    POLLUTANT_HEALTH,
    POLLUTANTS,
)

log = get_logger("emission_predictor")

FEATURES = [
    "total_count",
    "density",
    "avg_speed",
    "count_car",
    "count_motorcycle",
    "count_truck",
    "count_bus",
    "count_bicycle",
]


@dataclass
class EmissionEstimate:
    co2: float
    co: float
    nox: float
    pm25: float
    pm10: float
    hc: float
    voc: float
    so2: float
    ch4: float
    n2o: float
    co2e: float
    emission_score: float
    category: str
    source: str
    vehicle_breakdown: dict[str, dict[str, float]]
    gas_risk: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "co2": self.co2,
            "co": self.co,
            "nox": self.nox,
            "pm25": self.pm25,
            "pm10": self.pm10,
            "hc": self.hc,
            "voc": self.voc,
            "so2": self.so2,
            "ch4": self.ch4,
            "n2o": self.n2o,
            "co2e": self.co2e,
            "emission_score": self.emission_score,
            "category": self.category,
            "source": self.source,
            "vehicle_breakdown": self.vehicle_breakdown,
            "gas_risk": self.gas_risk,
        }


def emission_category(score: float) -> str:
    for label, lo, hi in config.EMISSION_BANDS:
        if lo <= score <= hi:
            return label
    return config.EMISSION_BANDS[-1][0]


class EmissionPredictor:
    def __init__(self, model_paths: dict[str, Path] | None = None) -> None:
        self.model_paths = model_paths or config.XGB_MODELS
        self._models: dict[str, object] | None = None

    def _load_models(self) -> dict[str, object] | None:
        if self._models is not None:
            return self._models
        if not all(Path(path).exists() for path in self.model_paths.values()):
            return None
        try:
            from xgboost import XGBRegressor

            models = {}
            for target, path in self.model_paths.items():
                model = XGBRegressor()
                model.load_model(str(path))
                models[target] = model
            self._models = models
            log.info("Loaded XGBoost emission models from %s", config.XGB_DIR)
            return models
        except Exception as exc:
            log.warning("Could not load XGBoost models; using fallback: %s", exc)
            return None

    @staticmethod
    def feature_vector(counts: dict[str, int], density: float, avg_speed: float) -> np.ndarray:
        total_count = int(sum(counts.values()))
        row = {
            "total_count": total_count,
            "density": float(density),
            "avg_speed": float(avg_speed),
            "count_car": int(counts.get("car", 0)),
            "count_motorcycle": int(counts.get("motorcycle", 0)),
            "count_truck": int(counts.get("truck", 0)),
            "count_bus": int(counts.get("bus", 0)),
            "count_bicycle": int(counts.get("bicycle", 0)),
        }
        return np.array([[row[name] for name in FEATURES]], dtype=np.float32)

    @staticmethod
    def _score(co2: float, nox: float, pm25: float) -> float:
        return float(np.clip(100.0 * (0.45 * co2 / 7000.0 + 0.35 * nox / 35.0 + 0.20 * pm25 / 1.5), 0, 100))

    @staticmethod
    def _speed_multiplier(avg_speed: float) -> float:
        """Urban stop-go correction: higher at very low speeds/idling."""
        speed = max(3.0, float(avg_speed))
        if speed < 12:
            return 1.45
        if speed < 25:
            return 1.25
        if speed > 65:
            return 1.10
        return 1.0

    @staticmethod
    def _density_multiplier(density: float) -> float:
        return 1.0 + min(0.35, max(0.0, float(density) - 35.0) / 200.0)

    def estimate_by_vehicle(
        self, counts: dict[str, int], density: float, avg_speed: float
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Return aggregate and vehicle-class emissions in g/km-equivalent."""
        multiplier = self._speed_multiplier(avg_speed) * self._density_multiplier(density)
        totals = {pollutant: 0.0 for pollutant in POLLUTANTS}
        breakdown: dict[str, dict[str, float]] = {}

        for vehicle_type, count in counts.items():
            count = max(0, int(count))
            factors = EMISSION_FACTORS_G_PER_KM.get(vehicle_type, {})
            row = {}
            for pollutant in POLLUTANTS:
                value = count * factors.get(pollutant, 0.0) * multiplier
                row[pollutant] = round(float(value), 4)
                totals[pollutant] += value
            row["count"] = count
            breakdown[vehicle_type] = row

        totals["co2e"] = (
            totals["co2"] * GWP_100["co2"]
            + totals["ch4"] * GWP_100["ch4"]
            + totals["n2o"] * GWP_100["n2o"]
        )
        for row in breakdown.values():
            row["co2e"] = round(
                row["co2"] * GWP_100["co2"]
                + row["ch4"] * GWP_100["ch4"]
                + row["n2o"] * GWP_100["n2o"],
                4,
            )
        return {key: round(float(value), 4) for key, value in totals.items()}, breakdown

    @staticmethod
    def _risk_scores(values: dict[str, float]) -> dict[str, float]:
        normalizers = {
            "co2": 7000.0,
            "co": 35.0,
            "nox": 35.0,
            "pm25": 1.5,
            "pm10": 2.5,
            "hc": 5.0,
            "voc": 4.0,
            "so2": 0.4,
            "ch4": 0.8,
            "n2o": 0.45,
            "co2e": 9000.0,
        }
        return {
            pollutant: round(float(np.clip(100.0 * values.get(pollutant, 0.0) / limit, 0, 100)), 2)
            for pollutant, limit in normalizers.items()
        }

    def _make_estimate(
        self,
        values: dict[str, float],
        breakdown: dict[str, dict[str, float]],
        source: str,
    ) -> EmissionEstimate:
        score = float(np.clip(
            0.22 * self._risk_scores(values)["co2e"]
            + 0.18 * self._risk_scores(values)["nox"]
            + 0.22 * self._risk_scores(values)["pm25"]
            + 0.16 * self._risk_scores(values)["pm10"]
            + 0.12 * self._risk_scores(values)["co"]
            + 0.10 * max(self._risk_scores(values)["voc"], self._risk_scores(values)["hc"]),
            0,
            100,
        ))
        return EmissionEstimate(
            co2=round(values.get("co2", 0.0), 2),
            co=round(values.get("co", 0.0), 3),
            nox=round(values.get("nox", 0.0), 3),
            pm25=round(values.get("pm25", 0.0), 4),
            pm10=round(values.get("pm10", 0.0), 4),
            hc=round(values.get("hc", 0.0), 3),
            voc=round(values.get("voc", 0.0), 3),
            so2=round(values.get("so2", 0.0), 4),
            ch4=round(values.get("ch4", 0.0), 4),
            n2o=round(values.get("n2o", 0.0), 4),
            co2e=round(values.get("co2e", 0.0), 2),
            emission_score=round(float(score), 2),
            category=emission_category(score),
            source=source,
            vehicle_breakdown=breakdown,
            gas_risk=self._risk_scores(values),
        )

    def _fallback(self, counts: dict[str, int], density: float, avg_speed: float) -> EmissionEstimate:
        values, breakdown = self.estimate_by_vehicle(counts, density, avg_speed)
        return self._make_estimate(values, breakdown, "factor_table")

    def predict(self, counts: dict[str, int], density: float, avg_speed: float) -> EmissionEstimate:
        models = self._load_models()
        if models is None:
            return self._fallback(counts, density, avg_speed)
        x = self.feature_vector(counts, density, avg_speed)
        values = {
            pollutant: max(0.0, float(models[pollutant].predict(x)[0]))
            for pollutant in config.EMISSION_POLLUTANTS
            if pollutant in models
        }
        factor_values, breakdown = self.estimate_by_vehicle(counts, density, avg_speed)
        for pollutant in config.EMISSION_POLLUTANTS:
            values.setdefault(pollutant, factor_values.get(pollutant, 0.0))
        return self._make_estimate(values, breakdown, "xgboost")


def pollutant_health_metadata() -> dict:
    return POLLUTANT_HEALTH
