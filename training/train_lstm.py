"""Train the LSTM congestion forecaster."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.device import resolve_device  # noqa: E402
from core.logger import get_logger  # noqa: E402
from prediction.traffic_predictor import FEATURES, TrafficLSTM  # noqa: E402

log = get_logger("train_lstm")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return df


def make_sequences(df: pd.DataFrame, sequence_length: int,
                   horizons: list[int]) -> tuple[np.ndarray, np.ndarray]:
    values = df[FEATURES].to_numpy(dtype=np.float32)
    target = df["congestion_score"].to_numpy(dtype=np.float32)
    x, y = [], []
    horizon_steps = [max(1, h // max(1, config.FORECAST_INTERVAL_MIN)) for h in horizons]
    max_h = max(horizon_steps)
    for i in range(sequence_length, len(df) - max_h):
        x.append(values[i - sequence_length:i])
        y.append([target[i + h] for h in horizon_steps])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train(args: argparse.Namespace) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run training/generate_sample_data.py first.")
    df = add_time_features(pd.read_csv(path))

    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    x, y = make_sequences(df, args.sequence_length, config.FORECAST_HORIZONS)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=False
    )

    device = resolve_device()
    wrapper = TrafficLSTM(
        input_size=len(FEATURES),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=len(config.FORECAST_HORIZONS),
    )
    model = wrapper.model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 5 == 0:
            log.info("Epoch %d/%d loss=%.4f", epoch, args.epochs, np.mean(losses))

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(x_test, device=device)).cpu().numpy()

    rmse = float(np.sqrt(mean_squared_error(y_test.reshape(-1), preds.reshape(-1))))
    mae = float(mean_absolute_error(y_test.reshape(-1), preds.reshape(-1)))
    r2 = float(r2_score(y_test.reshape(-1), preds.reshape(-1)))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "horizons": config.FORECAST_HORIZONS}

    config.LSTM_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": len(FEATURES),
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "output_size": len(config.FORECAST_HORIZONS),
        },
        config.LSTM_MODEL_PATH,
    )
    config.LSTM_SCALER_PATH.write_text(
        json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, indent=2),
        encoding="utf-8",
    )
    (config.DATA_DIR / "lstm_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM traffic forecaster")
    parser.add_argument("--csv", default=str(config.DATA_DIR / "traffic_history.csv"))
    parser.add_argument("--epochs", type=int, default=config.LSTM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.LSTM_BATCH_SIZE)
    parser.add_argument("--hidden-size", type=int, default=config.LSTM_HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=config.LSTM_NUM_LAYERS)
    parser.add_argument("--sequence-length", type=int, default=config.SEQUENCE_LENGTH)
    parser.add_argument("--lr", type=float, default=config.LSTM_LR)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
