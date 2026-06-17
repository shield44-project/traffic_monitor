# AI-Powered Traffic Congestion Prediction for Emission Reduction

**A Smart Traffic Intelligence System designed to mitigate urban pollution through predictive congestion management.**

This project detects and counts vehicles, fine-tunes YOLOv8 for emergency vehicles, **forecasts congestion using LSTM to enable proactive emission reduction**, and estimates gaseous pollutants with XGBoost.

## Primary Aim
To reduce the carbon and toxic footprint of urban traffic by predicting congestion peaks before they occur, allowing for proactive traffic flow management that keeps vehicles moving and minimizes idling emissions.

## Features

- Vehicle detection with pretrained YOLOv8n or YOLOv8s COCO weights.
- Emergency vehicle detection through YOLOv8 transfer learning.
- Vehicle counts, lane occupancy proxy, traffic density and congestion levels.
- LSTM predictions for 5, 10 and 15 minute congestion horizons.
- XGBoost/factor-table CO2, CO2e, CO, NOx, PM2.5, PM10, HC, VOC, SO2, CH4 and N2O emission prediction.
- Live video, webcam and RTSP support.
- Browser/phone camera frame capture and uploaded video analysis.
- User-installable vehicle traffic-density YOLO `.pt` models.
- Left/right lane ROI density analysis with Smooth/Heavy intensity labels.
- SQLite persistence, REST APIs, PDF/CSV reports.
- Dashboard with dark mode, Chart.js graphs, historical analytics and alerts.
- Authentication with admin/viewer roles.
- CPU fallback and CUDA GPU support when available.

## Quick Start

From the repository root:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

Default login:

```text
admin / admin123
```

Generate demo data for a populated dashboard:

```bash
cd traffic_ai_system
python training/generate_sample_data.py --days 7
```

## Project Structure

```text
traffic_ai_system/
  app.py
  config.py
  database.db
  database/
  detection/
  prediction/
  training/
  evaluation/
  models/
  datasets/
  reports/
  templates/
  static/
  tests/
```

## Model Training

Emergency YOLO fine-tuning:

```bash
cd traffic_ai_system
python training/preprocess_yolo_dataset.py --data datasets/emergency/data.yaml
python training/train_emergency_yolo.py --data datasets/emergency/data.yaml
```

The script starts from pretrained YOLOv8 weights, freezes the backbone for the
initial stage, fine-tunes detection layers, validates, and saves:

```text
models/yolo/emergency/best.pt
data/emergency_yolo_metrics.json
```

Install an existing trusted emergency `best.pt`:

```bash
python training/download_emergency_model.py --url "https://your-trusted-host/best.pt"
```

or upload `best.pt` from the Model Performance page. There is no official
Ultralytics emergency-vehicle checkpoint bundled with YOLOv8; the app does not
simulate emergency detections when `best.pt` is missing.

Install a vehicle/traffic-density `.pt` from the Model Performance page. The
app checks models in this order: `models/yolo/traffic_density_best.pt`, the
local `Smart-Traffic-Intelligence-System/best.pt`, the local
`YOLOv8_Traffic_Density_Estimation/models/best.pt`, then the Ultralytics YOLO
fallback.

Traffic forecasting:

```bash
python training/generate_sample_data.py --days 14
python training/train_lstm.py
```

Emission prediction:

```bash
python training/train_xgboost.py
```

## REST API

- `GET /api/summary`
- `POST /api/live/start`
- `POST /api/live/stop`
- `GET /api/live/status`
- `GET /api/live/frame`
- `POST /api/analyze-image`
- `POST /api/analyze-video`
- `POST /api/analyze-browser-frame`
- `POST /api/predict/traffic`
- `POST /api/predict/emissions`
- `GET /api/history/<table>`
- `GET|POST /api/cameras`

Authenticated session login is required for dashboard and API access.

## Database Tables

- `vehicles`
- `emergency_events`
- `traffic_data`
- `emissions`
- `predictions`
- `cameras`
- `users`

The schema is in `database/schema.sql`.

## Reports

PDF report:

```text
/reports -> Export PDF
```

CSV export supports `vehicles`, `emergency_events`, `traffic_data`, `emissions`
and `predictions`.

## Datasets

Supported datasets include UA-DETRAC, AI City Challenge, Roboflow emergency
vehicle datasets, and custom YOLO-format datasets. See `datasets/README.md`.

For a detailed literature review and report-ready reference list covering
YOLO traffic detection, ROI density, AI City Challenge, CityFlow, BDD100K,
BMD-45, LSTM forecasting and emissions estimation, see `PROJECT.md`.

## Testing

```bash
cd traffic_ai_system
pytest
```

## Notes

See `PROJECT.md` for the detailed objectives, methodology, literature review,
IEEE/paper directions, datasets used, testing-video sources, camera-feed flow,
model choices, public URL options, Docker usage and deployment guidance. See
`datasets/README.md` for dataset layout and recommended traffic video sources.

The dashboard remains usable before custom training by using transparent
fallbacks for LSTM and an EPA/MOVES/EEA-style emission factor table. Emissions
are engineering estimates in g/km-equivalent by detected vehicle class; exact
regulatory inventories require local MOVES/COPERT inputs such as fuel, vehicle
age, temperature, road grade and drive cycle. Emergency detection requires
`models/yolo/emergency/best.pt` for real ambulance/fire truck/police detection;
the fallback is only for interface demonstration.
