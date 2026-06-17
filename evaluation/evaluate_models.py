"""Evaluate available trained models and write metrics JSON files."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model evaluation scripts")
    parser.add_argument("--traffic-csv", default="data/traffic_history.csv")
    parser.add_argument("--emission-csv", default="data/emission_history.csv")
    parser.add_argument("--yolo-data", default=None)
    args = parser.parse_args()

    outputs = {}
    if Path(ROOT / args.traffic_csv).exists():
        run([sys.executable, "training/train_lstm.py", "--csv", args.traffic_csv, "--epochs", "3"])
        outputs["lstm"] = "data/lstm_metrics.json"
    if Path(ROOT / args.emission_csv).exists():
        run([sys.executable, "training/train_xgboost.py", "--csv", args.emission_csv])
        outputs["xgboost"] = "data/xgboost_metrics.json"
    if args.yolo_data:
        run([sys.executable, "training/train_emergency_yolo.py", "--data", args.yolo_data, "--freeze-epochs", "1", "--finetune-epochs", "1"])
        outputs["emergency_yolo"] = "data/emergency_yolo_metrics.json"
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
