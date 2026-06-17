"""
Module 3 - Traffic congestion analysis.

Turns a raw per-frame vehicle breakdown into interpretable traffic metrics:

* **density %**       - how full the road is (0-100), weighted by vehicle size
* **congestion score**- density adjusted for the share of heavy vehicles
* **level**           - Low / Medium / High / Severe band
* **avg speed**       - rough estimate (free-flow degraded by congestion)

The class is pure-Python/NumPy with no heavy dependencies so it is trivially
unit-testable and reusable from both live detection and batch analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import config


@dataclass
class CongestionResult:
    total_count: int
    density: float            # 0-100 %
    congestion_score: float   # 0-100
    level: str                # Low/Medium/High/Severe
    avg_speed: float          # km/h (estimated)

    def as_dict(self) -> dict:
        return asdict(self)


class CongestionAnalyzer:
    """Compute congestion metrics from vehicle-type counts."""

    def __init__(self, saturation_count: int | None = None,
                 weights: dict[str, float] | None = None,
                 bands=None):
        self.saturation = saturation_count or config.SATURATION_COUNT
        self.weights = weights or config.VEHICLE_WEIGHTS
        self.bands = bands or config.CONGESTION_BANDS

    # -- core math ---------------------------------------------------------
    def density(self, counts: dict[str, int]) -> float:
        """Weighted occupancy as a percentage of the saturation capacity."""
        weighted = sum(counts.get(t, 0) * self.weights.get(t, 1.0)
                       for t in counts)
        return float(min(100.0, 100.0 * weighted / max(1, self.saturation)))

    def _heavy_fraction(self, counts: dict[str, int]) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        heavy = counts.get("truck", 0) + counts.get("bus", 0)
        return heavy / total

    def classify(self, score: float) -> str:
        for label, lo, hi in self.bands:
            if lo <= score <= hi:
                return label
        return self.bands[-1][0]

    def estimate_speed(self, score: float) -> float:
        """Simple inverse relationship: higher congestion -> lower speed."""
        return float(max(5.0, 60.0 * (1.0 - score / 130.0)))

    # -- public API --------------------------------------------------------
    def analyze(self, counts: dict[str, int]) -> CongestionResult:
        """Compute all congestion metrics for one frame's vehicle counts."""
        counts = {k: int(v) for k, v in counts.items()}
        total = sum(counts.values())
        dens = self.density(counts)
        heavy = self._heavy_fraction(counts)
        score = float(min(100.0, dens * (1.0 + 0.25 * heavy)))
        level = self.classify(score)
        speed = self.estimate_speed(score)
        return CongestionResult(
            total_count=total, density=round(dens, 2),
            congestion_score=round(score, 2), level=level,
            avg_speed=round(speed, 2),
        )
