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
REFERENCE_TRAFFIC_DENSITY_MODEL = (
    BASE_DIR / "YOLOv8_Traffic_Density_Estimation" / "models" / "best.pt"
)
SMART_TRAFFIC_MODEL = BASE_DIR / "Smart-Traffic-Intelligence-System" / "best.pt"
TRAFFIC_DENSITY_MODEL = YOLO_DIR / "traffic_density_best.pt"

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

os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA_DIR / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(DATA_DIR / "matplotlib"))
(DATA_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "database.db"))
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------
def _default_vehicle_model() -> str:
    """Pick the safest local vehicle detector while preserving fallbacks.

    The one-class top-view reference model is useful for its original sample
    road video, but it is too broad for webcam use because it labels everything
    as a generic "Vehicle". Prefer a multi-class traffic checkpoint when it is
    present; otherwise use the standard COCO YOLO name so Ultralytics can fetch
    it when network access is available.
    """
    if TRAFFIC_DENSITY_MODEL.exists():
        return str(TRAFFIC_DENSITY_MODEL)
    if SMART_TRAFFIC_MODEL.exists():
        return str(SMART_TRAFFIC_MODEL)
    return str(YOLO_DIR / "yolov8n.pt")


# Vehicle detector. The preferred model is an installed traffic-density `.pt`;
# this repo then falls back to the included multi-class checkpoint or COCO
# weights. The one-class top-view model can still be installed explicitly.
VEHICLE_MODEL = os.getenv("VEHICLE_MODEL", _default_vehicle_model())
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

VEHICLE_CONF = _env_float("VEHICLE_CONF", 0.45)
VEHICLE_IOU = _env_float("VEHICLE_IOU", 0.45)
EMERGENCY_CONF = _env_float("EMERGENCY_CONF", 0.40)
INFER_IMGSZ = _env_int("INFER_IMGSZ", 640)
VIDEO_SAMPLE_EVERY_N_FRAMES = _env_int("VIDEO_SAMPLE_EVERY_N_FRAMES", 15)
VIDEO_MAX_ANALYSIS_FRAMES = _env_int("VIDEO_MAX_ANALYSIS_FRAMES", 120)
VIDEO_ANALYSIS_WIDTH = _env_int("VIDEO_ANALYSIS_WIDTH", 960)
ROI_HEAVY_TRAFFIC_THRESHOLD = _env_int("ROI_HEAVY_TRAFFIC_THRESHOLD", 10)
ROI_REFERENCE_WIDTH = _env_int("ROI_REFERENCE_WIDTH", 1280)
ROI_REFERENCE_HEIGHT = _env_int("ROI_REFERENCE_HEIGHT", 720)
ROI_VERTICAL_RANGE = (
    _env_int("ROI_VERTICAL_RANGE_TOP", 325),
    _env_int("ROI_VERTICAL_RANGE_BOTTOM", 635),
)
ROI_LANE_THRESHOLD_X = _env_int("ROI_LANE_THRESHOLD_X", 609)
ROI_LEFT_POLYGON = ((465, 350), (609, 350), (510, 630), (2, 630))
ROI_RIGHT_POLYGON = ((678, 350), (815, 350), (1203, 630), (743, 630))

# COCO class ids that count as "vehicles" mapped to friendly names.
# (Ids come from the standard COCO dataset used by pretrained YOLOv8.)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}
VEHICLE_TYPES = ("car", "motorcycle", "bus", "truck", "bicycle", "vehicle")
VEHICLE_LABEL_ALIASES = {
    "car": "car",
    "cars": "car",
    "mobil": "car",
    "motor": "motorcycle",
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "bikes": "motorcycle",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "bus": "bus",
    "buses": "bus",
    "truck": "truck",
    "truk": "truck",
    "lorry": "truck",
    "bicycle": "bicycle",
    "vehicle": "vehicle",
    "vehicles": "vehicle",
}
GENERIC_VEHICLE_LABELS = {"vehicle", "vehicles", "auto", "traffic"}
CONCRETE_VEHICLE_LABELS = set(VEHICLE_LABEL_ALIASES) - GENERIC_VEHICLE_LABELS

# Precision-first filtering for live monitoring. These thresholds reduce
# webcam false positives, especially when a broad custom checkpoint is present.
VEHICLE_TYPE_MIN_CONFIDENCE = {
    "car": _env_float("VEHICLE_CAR_CONF", 0.45),
    "motorcycle": _env_float("VEHICLE_MOTORCYCLE_CONF", 0.42),
    "bus": _env_float("VEHICLE_BUS_CONF", 0.42),
    "truck": _env_float("VEHICLE_TRUCK_CONF", 0.42),
    "bicycle": _env_float("VEHICLE_BICYCLE_CONF", 0.40),
    "vehicle": _env_float("GENERIC_VEHICLE_CONF", 0.70),
}
ALLOW_GENERIC_VEHICLE_CLASS = _env_bool("ALLOW_GENERIC_VEHICLE_CLASS", False)
VEHICLE_MIN_BOX_AREA_RATIO = _env_float("VEHICLE_MIN_BOX_AREA_RATIO", 0.00015)
VEHICLE_MAX_BOX_AREA_RATIO = _env_float("VEHICLE_MAX_BOX_AREA_RATIO", 0.70)
GENERIC_VEHICLE_MAX_BOX_AREA_RATIO = _env_float("GENERIC_VEHICLE_MAX_BOX_AREA_RATIO", 0.24)
VEHICLE_MIN_ASPECT_RATIO = _env_float("VEHICLE_MIN_ASPECT_RATIO", 0.18)
VEHICLE_MAX_ASPECT_RATIO = _env_float("VEHICLE_MAX_ASPECT_RATIO", 6.50)
GENERIC_VEHICLE_MIN_ASPECT_RATIO = _env_float("GENERIC_VEHICLE_MIN_ASPECT_RATIO", 0.35)
GENERIC_VEHICLE_MAX_ASPECT_RATIO = _env_float("GENERIC_VEHICLE_MAX_ASPECT_RATIO", 4.50)

SAMPLE_VIDEO_SOURCES = tuple(
    str(path.relative_to(BASE_DIR))
    for path in (
        UPLOAD_DIR / "sample_video.mp4",
        UPLOAD_DIR / "18437773-uhd_3840_2160_50fps.mp4",
        BASE_DIR / "YOLOv8_Traffic_Density_Estimation" / "sample_video.mp4",
    )
    if path.exists()
)

# Emergency classes produced by the fine-tuned model (index order must match
# the training data.yaml `names:` list).
EMERGENCY_CLASSES = ["ambulance", "fire_truck", "police"]
EMERGENCY_LABEL_ALIASES = {
    "ambulance": "ambulance",
    "ambulances": "ambulance",
    "fire_truck": "fire_truck",
    "firetruck": "fire_truck",
    "fire_engine": "fire_truck",
    "fireengine": "fire_truck",
    "fire": "fire_truck",
    "police": "police",
    "police_car": "police",
    "policecar": "police",
    "patrol_car": "police",
}

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
    "vehicle": 1.2,
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
