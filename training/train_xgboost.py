"""Train XGBoost regressors for traffic-related pollutant emissions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.logger import get_logger  # noqa: E402
from prediction.emission_predictor import FEATURES  # noqa: E402

log = get_logger("train_xgboost")


def train(args: argparse.Namespace) -> dict:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run training/generate_sample_data.py first.")
    df = pd.read_csv(csv_path)
    x = df[FEATURES].to_numpy(dtype=np.float32)
    metrics = {}

    config.XGB_DIR.mkdir(parents=True, exist_ok=True)
    for target in config.EMISSION_POLLUTANTS:
        y = df[target].to_numpy(dtype=np.float32)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        )
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        target_metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds)),
        }
        metrics[target] = target_metrics
        model.save_model(str(config.XGB_MODELS[target]))
        log.info("%s metrics: %s", target, target_metrics)

    (config.DATA_DIR / "xgboost_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost emission models")
    parser.add_argument("--csv", default=str(config.DATA_DIR / "emission_history.csv"))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
