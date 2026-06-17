"""
Central configuration for the AI-Powered Smart Traffic System.

Everything tunable lives here: filesystem paths, model locations, detection
thresholds, congestion bands, forecasting horizons and Flask settings. Values
can be overridden through environment variables (see ``.env.example``) so the
same code runs unchanged on a laptop, a server or inside Docker.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # keeps tests/imports usable before dependencies are installed
    def load_dotenv(*_args, **_kwargs):
        return False

# Load variables from a local .env file if present (no-op otherwise).
load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
YOLO_DIR = MODELS_DIR / "yolo"
LSTM_DIR = MODELS_DIR / "lstm"
XGB_DIR = MODELS_DIR / "xgboost"

DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = BASE_DIR / "datasets"
REPORTS_DIR = BASE_DIR / "reports" / "generated"
SAMPLES_DIR = BASE_DIR / "samples"
LOGS_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = STATIC_DIR / "uploads"
FRAME_DIR = STATIC_DIR / "frames"

# Ensure runtime directories exist (cheap, idempotent).
for _d in (MODELS_DIR, YOLO_DIR, YOLO_DIR / "emergency", LSTM_DIR, XGB_DIR,
           DATA_DIR, REPORTS_DIR, SAMPLES_DIR, LOGS_DIR, UPLOAD_DIR, FRAME_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "database.db"))
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------
# Pretrained base model for general vehicle detection (auto-downloaded by
# Ultralytics on first use from: https://github.com/ultralytics/assets/releases)
# Use yolov8s.pt or yolov8m.pt for higher accuracy on a GPU.
VEHICLE_MODEL = os.getenv("VEHICLE_MODEL", str(YOLO_DIR / "yolov8n.pt"))
# Fine-tuned weights for emergency vehicles (produced by training script).
EMERGENCY_MODEL = os.getenv("EMERGENCY_MODEL", str(YOLO_DIR / "emergency" / "best.pt"))

LSTM_MODEL_PATH = LSTM_DIR / "lstm_traffic.pt"
LSTM_SCALER_PATH = LSTM_DIR / "scaler.json"

XGB_MODELS = {
    "co2": XGB_DIR / "co2.json",
    "co": XGB_DIR / "co.json",
    "nox": XGB_DIR / "nox.json",
    "pm25": XGB_DIR / "pm25.json",
    "pm10": XGB_DIR / "pm10.json",
    "hc": XGB_DIR / "hc.json",
    "voc": XGB_DIR / "voc.json",
    "so2": XGB_DIR / "so2.json",
    "ch4": XGB_DIR / "ch4.json",
    "n2o": XGB_DIR / "n2o.json",
    "co2e": XGB_DIR / "co2e.json",
}

EMISSION_POLLUTANTS = [
    "co2", "co", "nox", "pm25", "pm10", "hc", "voc", "so2", "ch4", "n2o", "co2e"
]

# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------
# Device is resolved at runtime in core/device.py; "auto" picks GPU if present.
DEVICE = os.getenv("DEVICE", "auto")

VEHICLE_CONF = _env_float("VEHICLE_CONF", 0.35)
VEHICLE_IOU = _env_float("VEHICLE_IOU", 0.45)
EMERGENCY_CONF = _env_float("EMERGENCY_CONF", 0.40)
INFER_IMGSZ = _env_int("INFER_IMGSZ", 640)
VIDEO_SAMPLE_EVERY_N_FRAMES = _env_int("VIDEO_SAMPLE_EVERY_N_FRAMES", 15)
VIDEO_MAX_ANALYSIS_FRAMES = _env_int("VIDEO_MAX_ANALYSIS_FRAMES", 120)

# COCO class ids that count as "vehicles" mapped to friendly names.
# (Ids come from the standard COCO dataset used by pretrained YOLOv8.)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}

# Emergency classes produced by the fine-tuned model (index order must match
# the training data.yaml `names:` list).
EMERGENCY_CLASSES = ["ambulance", "fire_truck", "police"]

# ---------------------------------------------------------------------------
# Congestion model
# ---------------------------------------------------------------------------
# Vehicle count that represents a fully saturated frame (100% density). Used to
# normalise raw counts into a 0-100 density percentage.
SATURATION_COUNT = _env_int("SATURATION_COUNT", 40)

# Relative road footprint of each vehicle type (cars = 1.0 baseline). Heavier
# vehicles occupy more lane space, so they push density up faster.
VEHICLE_WEIGHTS = {
    "car": 1.0,
    "motorcycle": 0.5,
    "bicycle": 0.4,
    "bus": 2.5,
    "truck": 2.2,
}

# Congestion bands: (label, lower_bound_inclusive, upper_bound_inclusive)
CONGESTION_BANDS = [
    ("Low", 0, 25),
    ("Medium", 26, 50),
    ("High", 51, 75),
    ("Severe", 76, 100),
]

# ---------------------------------------------------------------------------
# Forecasting (LSTM)
# ---------------------------------------------------------------------------
# Each data point represents one aggregation interval (default 1 minute).
FORECAST_INTERVAL_MIN = _env_int("FORECAST_INTERVAL_MIN", 1)
# Horizons (in minutes) to predict.
FORECAST_HORIZONS = [5, 10, 15]
# Number of past intervals fed into the LSTM as one input sequence.
SEQUENCE_LENGTH = _env_int("SEQUENCE_LENGTH", 20)
LSTM_HIDDEN_SIZE = _env_int("LSTM_HIDDEN_SIZE", 64)
LSTM_NUM_LAYERS = _env_int("LSTM_NUM_LAYERS", 2)
LSTM_EPOCHS = _env_int("LSTM_EPOCHS", 40)
LSTM_BATCH_SIZE = _env_int("LSTM_BATCH_SIZE", 32)
LSTM_LR = _env_float("LSTM_LR", 1e-3)

# ---------------------------------------------------------------------------
# Emission model (XGBoost)
# ---------------------------------------------------------------------------
# Pollution category thresholds keyed on the composite emission score (0-100).
EMISSION_BANDS = [
    ("Good", 0, 25),
    ("Moderate", 26, 50),
    ("Unhealthy", 51, 75),
    ("Hazardous", 76, 100),
]

# ---------------------------------------------------------------------------
# Flask / web
# ---------------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _env_int("PORT", 5000)
DEBUG = _env_bool("DEBUG", False)

# Seed admin account (created on first DB init if no users exist).
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Roles allowed in the system.
ROLES = ("admin", "viewer")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "traffic_ai.log"
