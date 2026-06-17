"""
Generate realistic **synthetic** traffic & emission data.

Why this exists
---------------
The project ships with no real dataset and the dev laptop has no GPU, yet the
LSTM forecaster and XGBoost emission models need data to learn from. This script
fabricates a believable multi-day time series with the structure real urban
traffic has — morning/evening rush peaks, quieter nights, lighter weekends,
plus noise — so the downstream models train end-to-end in minutes.

Outputs
-------
* ``data/traffic_history.csv``   - per-interval traffic + congestion features
* ``data/emission_history.csv``  - features + pollutant targets
* Rows inserted into SQLite (vehicles, traffic_data, emissions,
  emergency_events) so the dashboard has something to show immediately.

Run:  python training/generate_sample_data.py [--days 7] [--no-db]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Allow "python training/generate_sample_data.py" from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.logger import get_logger  # noqa: E402
from prediction.emission_predictor import EmissionPredictor  # noqa: E402

log = get_logger("gen_data")

# Deterministic output so results are reproducible for a report.
RNG = np.random.default_rng(42)

# Mean share of each vehicle type in a typical frame (sums to 1.0).
TYPE_MIX = {
    "car": 0.62,
    "motorcycle": 0.18,
    "truck": 0.08,
    "bus": 0.05,
    "bicycle": 0.07,
}

EMISSION_MODEL = EmissionPredictor(
    model_paths={pollutant: Path("__missing__") for pollutant in config.EMISSION_POLLUTANTS}
)


def band_label(value: float, bands) -> str:
    """Map a 0-100 value to its band label using config band definitions."""
    for label, lo, hi in bands:
        if lo <= value <= hi:
            return label
    return bands[-1][0]


def _time_of_day_factor(hour: float) -> float:
    """Two Gaussian rush-hour bumps (~9am, ~6pm) on a low night baseline."""
    morning = math.exp(-((hour - 9.0) ** 2) / (2 * 1.6 ** 2))
    evening = math.exp(-((hour - 18.0) ** 2) / (2 * 1.8 ** 2))
    midday = 0.45 * math.exp(-((hour - 13.0) ** 2) / (2 * 2.5 ** 2))
    return 0.08 + 0.95 * morning + 1.0 * evening + midday


def generate_dataframe(days: int, interval_min: int) -> pd.DataFrame:
    """Build the full synthetic dataset as a DataFrame."""
    steps = int(days * 24 * 60 / interval_min)
    start = datetime.now(timezone.utc) - timedelta(minutes=steps * interval_min)

    records = []
    # A slowly drifting "momentum" term so congestion is autocorrelated rather
    # than pure noise (this is what makes the LSTM forecast worthwhile).
    momentum = 0.0
    for i in range(steps):
        ts = start + timedelta(minutes=i * interval_min)
        hour = ts.hour + ts.minute / 60.0
        is_weekend = ts.weekday() >= 5

        tod = _time_of_day_factor(hour)
        weekend_factor = 0.7 if is_weekend else 1.0

        # Base expected count for this interval.
        base = config.SATURATION_COUNT * 0.9 * tod * weekend_factor
        noise = RNG.normal(0, config.SATURATION_COUNT * 0.06)
        momentum = 0.85 * momentum + 0.15 * noise  # AR(1)-style smoothing
        total = max(0, base + momentum + noise * 0.5)

        # Split into vehicle types with a little per-interval variation.
        mix = {k: max(0.0, v + RNG.normal(0, 0.02)) for k, v in TYPE_MIX.items()}
        mix_sum = sum(mix.values())
        counts = {k: int(round(total * v / mix_sum)) for k, v in mix.items()}
        total_count = sum(counts.values())

        # Weighted density (heavy vehicles occupy more road).
        weighted = sum(counts[t] * config.VEHICLE_WEIGHTS[t] for t in counts)
        density = min(100.0, 100.0 * weighted / config.SATURATION_COUNT)

        # Congestion score: density nudged by how "heavy" the mix is.
        heavy_frac = (counts["truck"] + counts["bus"]) / max(1, total_count)
        score = min(100.0, density * (1.0 + 0.25 * heavy_frac))
        level = band_label(score, config.CONGESTION_BANDS)

        # Average speed falls as congestion rises (free-flow ~60 km/h).
        avg_speed = max(5.0, 60.0 * (1.0 - score / 130.0) + RNG.normal(0, 2.5))

        emissions = EMISSION_MODEL.predict(counts, density, avg_speed).as_dict()

        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": hour,
            "dow": ts.weekday(),
            "is_weekend": int(is_weekend),
            "total_count": total_count,
            **{f"count_{k}": counts[k] for k in counts},
            "density": round(density, 2),
            "congestion_score": round(score, 2),
            "congestion_level": level,
            "avg_speed": round(avg_speed, 2),
            "heavy_frac": round(heavy_frac, 4),
            **{key: emissions[key] for key in config.EMISSION_POLLUTANTS},
            "emission_score": emissions["emission_score"],
            "emission_category": emissions["category"],
            "gas_risk_json": json_dumps(emissions["gas_risk"]),
            "vehicle_breakdown_json": json_dumps(emissions["vehicle_breakdown"]),
        })

    return pd.DataFrame.from_records(records)


def write_csvs(df: pd.DataFrame) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic_cols = ["timestamp", "hour", "dow", "is_weekend", "total_count",
                    "count_car", "count_motorcycle", "count_truck",
                    "count_bus", "count_bicycle", "density",
                    "congestion_score", "congestion_level", "avg_speed"]
    emission_cols = ["timestamp", "total_count", "density", "avg_speed",
                     "heavy_frac", "count_car", "count_motorcycle",
                     "count_truck", "count_bus", "count_bicycle",
                     *config.EMISSION_POLLUTANTS, "emission_score",
                     "emission_category", "gas_risk_json",
                     "vehicle_breakdown_json"]
    df[traffic_cols].to_csv(config.DATA_DIR / "traffic_history.csv", index=False)
    df[emission_cols].to_csv(config.DATA_DIR / "emission_history.csv",
                             index=False)
    log.info("Wrote CSVs to %s", config.DATA_DIR)


def write_to_db(df: pd.DataFrame, emergency_rate: float = 0.004) -> None:
    """Insert a (down-sampled) view of the data into SQLite for the dashboard."""
    from database import db
    db.init_db(seed_admin=True)

    cameras = db.fetch_cameras()
    camera_id = cameras[0]["id"] if cameras else None

    # Insert every row's aggregate snapshots; vehicles only every 5th row to
    # keep the table from exploding while still being representative.
    n_emerg = 0
    for idx, row in df.iterrows():
        ts = row["timestamp"]
        db.insert_traffic_data(
            total_count=int(row["total_count"]), density=row["density"],
            score=row["congestion_score"], level=row["congestion_level"],
            avg_speed=row["avg_speed"], camera_id=camera_id, timestamp=ts)
        db.insert_emission(
            co2=row["co2"], co=row["co"], nox=row["nox"], pm25=row["pm25"],
            pm10=row["pm10"], hc=row["hc"], voc=row["voc"], so2=row["so2"],
            ch4=row["ch4"], n2o=row["n2o"], co2e=row["co2e"],
            score=row["emission_score"], category=row["emission_category"],
            gas_risk=json.loads(row["gas_risk_json"]),
            vehicle_breakdown=json.loads(row["vehicle_breakdown_json"]),
            camera_id=camera_id, timestamp=ts)
        if idx % 5 == 0:
            db.insert_vehicle_counts(
                {k: int(row[f"count_{k}"]) for k in TYPE_MIX},
                camera_id=camera_id, timestamp=ts)
        # Sprinkle emergency events, more likely during heavy congestion.
        if RNG.random() < emergency_rate * (1 + row["congestion_score"] / 100):
            vtype = RNG.choice(config.EMERGENCY_CLASSES)
            db.insert_emergency_event(
                vehicle_type=str(vtype),
                confidence=float(RNG.uniform(0.55, 0.97)),
                camera_id=camera_id, timestamp=ts)
            n_emerg += 1

    log.info("Inserted %d snapshots and %d emergency events into DB",
             len(df), n_emerg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic traffic data")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days of data to generate (default 7)")
    parser.add_argument("--interval", type=int,
                        default=config.FORECAST_INTERVAL_MIN,
                        help="Aggregation interval in minutes (default 1)")
    parser.add_argument("--no-db", action="store_true",
                        help="Only write CSVs, skip database insertion")
    args = parser.parse_args()

    log.info("Generating %d day(s) of synthetic data at %d-min intervals...",
             args.days, args.interval)
    df = generate_dataframe(args.days, args.interval)
    log.info("Generated %d rows", len(df))

    write_csvs(df)
    if not args.no_db:
        write_to_db(df)

    print("\n=== Sample data summary ===")
    print(f"Rows                : {len(df)}")
    print(f"Mean total_count    : {df['total_count'].mean():.1f}")
    print(f"Mean congestion     : {df['congestion_score'].mean():.1f}")
    print("Congestion levels   :")
    print(df["congestion_level"].value_counts().to_string())
    print(f"\nCSVs written to     : {config.DATA_DIR}")
    if not args.no_db:
        print(f"Database populated   : {config.DATABASE_PATH}")


def json_dumps(value: dict) -> str:
    return json.dumps(value, separators=(",", ":"))


if __name__ == "__main__":
    main()
