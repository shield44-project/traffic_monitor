# Project Aim: Traffic Congestion Prediction for Emission Reduction

## Primary Objective

The main aim of this project is **to mitigate the environmental impact of urban transportation by predicting traffic congestion peaks before they occur.** By utilizing **LSTM-based time-series forecasting**, the system enables proactive traffic management strategies that maintain smooth traffic flow, which is significantly more fuel-efficient and less polluting than congested "stop-and-go" conditions.

## Technical Goals

-   **Predictive Management:** Use LSTM networks to forecast traffic congestion horizons (5, 10, 15 min), allowing for preemptive intervention.
-   **Pollution Quantification:** Employ XGBoost regression and EPA-anchored factor tables to provide high-fidelity estimates of 11 gaseous pollutants (CO2, NOx, PM2.5, etc.).
-   **Real-Time Monitoring:** Deploy YOLOv8 for sub-second vehicle detection and emergency vehicle identification.
-   **Traffic Analysis:** Convert visual camera feeds into actionable metrics including density, average speed, and lane-specific intensity.

## What Was Fixed And Added

- Added support for traffic-density YOLO `.pt` checkpoints from the local reference projects:
  - `YOLOv8_Traffic_Density_Estimation/models/best.pt`
  - `Smart-Traffic-Intelligence-System/best.pt`
- Added an admin install flow for a custom vehicle/traffic YOLO `.pt` model from upload or trusted URL.
- Kept the emergency YOLO `.pt` install flow for ambulance/fire truck/police detection.
- Added left/right lane ROI density analysis based on `YOLOv8_Traffic_Density_Estimation`.
- Added custom label mapping for models that output `mobil`, `motor`, `truk`, or generic `vehicle`.
- Added live lane markers, lane counts, lane intensity, emergency markers, congestion, CO2e and fuel-waste estimates.
- Made uploaded videos more flexible. The app now accepts any file extension and lets OpenCV/FFmpeg decide whether frames can be decoded.
- Improved browser/phone camera handling. Browser camera frames are analyzed by the same backend and update the live dashboard.
- Fixed missing emergency-model behavior so live/video analysis does not crash when emergency `best.pt` is absent.
- Reworked the Live and Models UI with a glassmorphism style and user-friendly controls.
- Added video report details: preview frames, timeline, peak frame, lane balance, CO2e and fuel waste.

## Reference Projects Used

### YOLOv8_Traffic_Density_Estimation

Used for:

- Fine-tuned YOLOv8 traffic-density `.pt` model.
- Left and right road-region polygons.
- Per-lane vehicle counting.
- Smooth/Heavy traffic-intensity labeling.

The original script uses two quadrilateral regions and a vertical analysis band. This project adapts those ideas into reusable backend detector logic and overlays them on live/uploaded frames.

### Smart-Traffic-Intelligence-System

Used for:

- A second available `best.pt` fallback.
- User-friendly report concepts: total vehicles, density, pollution/impact and fuel waste.
- Historical/presentation style ideas.

This Flask app keeps its existing database, emissions, reports, auth and live dashboard instead of adding a separate Streamlit app.

## Literature Review

The project combines ideas from traffic object detection, multi-camera traffic analytics, traffic forecasting, emergency vehicle detection and emission estimation. IEEE Xplore blocks automated access in some environments, so the list below uses open paper pages and official dataset/challenge pages where available. For a final university report, verify DOI/page numbers from IEEE Xplore, IEEE/CVF proceedings, Springer/ECCV proceedings or Google Scholar before formatting the bibliography.

### Papers And Sources Referred

| Year | Paper / Source | Why It Is Used In This Project | Link |
| --- | --- | --- | --- |
| 2026 | Sharma et al., "BMD-45: A Large-Scale CCTV Vehicle Detection Dataset for Urban Traffic in Developing Cities" | Latest CCTV traffic-detection dataset direction; useful for dense, heterogeneous city traffic and Indian-style road scenes. Accepted for IEEE/CVF CVPR 2026 Findings according to the dataset/paper page. | `https://arxiv.org/abs/2604.24419` |
| 2026 | BMD-45 Hugging Face dataset card | Practical dataset access and statistics for CCTV vehicle detection: 45,986 images, about 481,947 boxes, 14 vehicle classes. | `https://huggingface.co/datasets/iisc-aim/BMD-45` |
| 2026 | Ullah et al., "Attention-Augmented YOLOv8 with Ghost Convolution for Real-Time Vehicle Detection in Intelligent Transportation Systems" | Recent YOLOv8 vehicle-detection architecture direction for ITS. Supports why YOLO is suitable for real-time traffic detection. | `https://arxiv.org/abs/2604.22856` |
| 2024 | Wang et al., "The 8th AI City Challenge" | Current benchmark context for intelligent traffic systems, traffic safety video understanding and large-scale video analytics. | `https://arxiv.org/abs/2404.09432` |
| 2026 | AI City Challenge official website | Current challenge/dataset direction for intelligent transportation, traffic anomaly reasoning, traffic safety captioning and cross-city object detection. | `https://www.aicitychallenge.org/` |
| 2024 | Khalili and Smyth, "SOD-YOLOv8 -- Enhancing YOLOv8 for Small Object Detection in Traffic Scenes" | Supports the small-object issue in traffic scenes, where far CCTV vehicles are hard to detect. | `https://arxiv.org/abs/2408.04786` |
| 2024 | Goenawan, "ASTM: Autonomous Smart Traffic Management System Using Artificial Intelligence CNN and LSTM" | Similar AI traffic-management idea using vehicle detection plus LSTM forecasting. Supports this project combining YOLO-style detection and LSTM prediction. | `https://arxiv.org/abs/2410.10929` |
| 2025 | Al Mhdawi et al., "Connecting Vision and Emissions: A Behavioural AI Approach to Carbon Estimation in Road Design" | Supports connecting vehicle computer vision with carbon/emission estimation. | `https://arxiv.org/abs/2506.18924` |
| 2019 | Tang et al., "CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification" | Standard city-scale traffic-camera benchmark for future multi-camera expansion. | `https://arxiv.org/abs/1903.09254` |
| 2018/2020 | Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning" | Large driving-video dataset for testing robustness across weather, location, time and road types. | `https://arxiv.org/abs/1805.04687` |
| 1997 | Hochreiter and Schmidhuber, "Long Short-Term Memory" | Foundational LSTM model used for traffic forecasting. | Use IEEE/Google Scholar/library citation lookup |
| Current | Ultralytics YOLO documentation/model releases | Practical `.pt` model loading, prediction and deployment path used by the app. | `https://docs.ultralytics.com/` |
| Current | EPA MOVES / EEA emission-factor methodology | Supports factor-based fallback emission estimation when trained XGBoost models are missing. | `https://www.epa.gov/moves` |

### Object Detection For Traffic Monitoring

YOLO-style one-stage detectors are appropriate because traffic monitoring needs a practical speed/accuracy balance. This app loads `.pt` models with Ultralytics and runs inference frame-by-frame. Recent traffic-scene work such as SOD-YOLOv8 and attention-augmented YOLOv8 shows that small distant vehicles, occlusion and CCTV viewpoint variation are major challenges.

How it maps to this project:

- YOLO detects cars, motorcycles, buses, trucks, bicycles or generic vehicles.
- The detector supports COCO-pretrained weights and custom traffic `.pt` checkpoints.
- Detections are converted into structured counts, boxes, lane density and emission inputs.

### Traffic Density And Region-Of-Interest Counting

`YOLOv8_Traffic_Density_Estimation` demonstrates a practical ROI-based traffic-density workflow with left/right polygon regions and a heavy-traffic threshold. AI City Challenge and CityFlow show why city-scale video analytics often need camera-specific regions, tracking and domain-specific datasets.

How it maps to this project:

- The app draws two road ROIs.
- Every detected vehicle center is assigned to left or right lane.
- Each lane gets count and Smooth/Heavy intensity.
- The live view shows lane markers and current lane state.

### Emergency Vehicle Detection

Emergency vehicle priority systems commonly use visual object detection, siren/acoustic detection or V2X communication. This project uses the visual-detection route because it fits the same YOLO frame-analysis pipeline used for general traffic monitoring.

How it maps to this project:

- Emergency detection is handled by a separate YOLO `.pt`.
- Expected classes are `ambulance`, `fire_truck`, `police`.
- If the emergency model is missing, the app reports that status instead of fabricating detections.

### Traffic Flow Forecasting

The LSTM component is based on the standard sequence-learning idea from Hochreiter and Schmidhuber. The ASTM traffic-management paper is a close project-level reference because it combines visual vehicle detection and LSTM-style traffic forecasting.

How it maps to this project:

- Recent traffic snapshots are passed to the predictor.
- The app forecasts 5, 10 and 15 minute congestion.
- If trained LSTM files are missing, a baseline forecast still keeps the UI usable.

### Vehicle Emission Estimation

The emissions component follows two ideas: engineering factor tables from EPA MOVES/EEA-style methods, and recent vision-to-emission research that connects detected vehicle classes with carbon estimates.

How it maps to this project:

- Detected vehicle counts are mapped into pollutant estimates.
- XGBoost models are used when trained files exist.
- A factor-table fallback estimates CO2, NOx, PM, CO, VOC, CH4, N2O and CO2e.
- Fuel-waste is estimated from idle time by congestion level.

### Dataset And Benchmark Literature

- BDD100K: large driving-video benchmark for heterogeneous road scenes.
- UA-DETRAC: classic vehicle detection/tracking traffic-camera benchmark.
- AI City Challenge / CityFlow: multi-camera city traffic analytics benchmark.
- BMD-45: newer CCTV vehicle dataset direction for realistic road monitoring.
- Roboflow/Kaggle top-view vehicle datasets: practical YOLO-format datasets for student training.

Use these sources to justify why the system is built around YOLO detection, video frame sampling, ROI density analysis, forecasting and emission estimation.

## AI Models Used

### Vehicle / Traffic Density YOLO

Primary model priority:

1. `models/yolo/traffic_density_best.pt` if installed from the Models page.
2. `Smart-Traffic-Intelligence-System/best.pt` if available.
3. `YOLOv8_Traffic_Density_Estimation/models/best.pt` if available.
4. `models/yolo/yolov8n.pt` fallback, which Ultralytics can download from its GitHub assets when network access is available.

Supported labels include COCO classes like `car`, `motorcycle`, `bus`, `truck`, `bicycle`, plus custom labels like `mobil`, `motor`, `truk` and generic `vehicle`.

### Emergency Vehicle YOLO

Expected path:

```text
models/yolo/emergency/best.pt
```

Expected classes:

```text
ambulance, fire_truck, police
```

The app does not fake emergency detections if this model is missing.

### Congestion Model

The congestion analyzer converts vehicle counts into:

- density percentage,
- congestion score,
- Low/Medium/High/Severe level,
- estimated average speed.

Heavy vehicles like buses and trucks receive higher density weights.

### Emission Model

The app estimates:

- CO2,
- CO,
- NOx,
- PM2.5,
- PM10,
- HC,
- VOC,
- SO2,
- CH4,
- N2O,
- CO2e.

It uses trained XGBoost models if present. Otherwise it uses a documented factor-table fallback by vehicle class. Generic `vehicle` detections also receive a default mixed-traffic emission factor.

### Traffic Forecasting

The LSTM predictor forecasts future congestion for 5, 10 and 15 minutes. If trained LSTM files are missing, the app uses a transparent baseline from recent traffic records.

## Dataset Notes

The project can use:

- COCO-pretrained YOLO vehicle classes.
- Top-view vehicle detection dataset referenced by `YOLOv8_Traffic_Density_Estimation`.
- Roboflow/Kaggle custom YOLO traffic datasets.
- Custom emergency vehicle datasets with `ambulance`, `fire_truck`, `police`.
- Generated sample traffic data from `training/generate_sample_data.py`.

Dataset documentation is also in `datasets/README.md`.

## Datasets Used Or Recommended

### Included / Local

- `static/uploads/cctv052x2004080516x01638.avi`: sample AVI file already present in this repo for quick upload/live-file testing.
- `YOLOv8_Traffic_Density_Estimation/sample_video.mp4`: sample traffic video from the reference project.
- `YOLOv8_Traffic_Density_Estimation/models/best.pt`: traffic-density YOLO model from the reference project.
- `Smart-Traffic-Intelligence-System/best.pt`: alternate vehicle model from the second reference project.
- `training/generate_sample_data.py`: creates synthetic historical traffic/emission records for dashboard testing.

### Public Video/Test Datasets To Try

1. AI City Challenge
   - Best for: real traffic-camera videos, multi-camera experiments, city analytics.
   - Use: download a challenge split, pick short `.mp4` clips, upload through Live -> Video Analysis.
   - Link: `https://www.aicitychallenge.org/`

2. UA-DETRAC
   - Best for: fixed traffic-camera vehicle detection and tracking.
   - Use: convert sequence/video clips to `.mp4` or `.avi`, then upload or use as server source.
   - Search: "UA-DETRAC dataset official".

3. BDD100K
   - Best for: dashcam road-scene videos with diverse weather/time conditions.
   - Use: short video clips for general vehicle detection testing.
   - Search: "BDD100K official download".

4. CityFlow / AI City Challenge datasets
   - Best for: multi-camera city intersections and vehicle tracking/re-identification.
   - Use: choose one camera stream at a time for this Flask app.
   - Search: "CityFlow dataset AI City Challenge".

5. Roboflow Top-View Vehicle Detection
   - Best for: YOLO-format vehicle training images.
   - Use: train/fine-tune a `.pt`, then install it from Models -> Install Vehicle YOLO .pt.
   - Search: "Roboflow Top-View Vehicle Detection YOLOv8".

6. Kaggle traffic videos or Pexels traffic videos
   - Best for: quick demo clips when official datasets are too large.
   - Use: download short clips, prefer 720p or lower for CPU inference.
   - Note: confirm license before publishing results.

### Recommended Test Clips

Use a mix of:

- low-density road clip,
- medium traffic junction clip,
- heavy congestion clip,
- night/low-light clip,
- rainy or blurry clip,
- clip with buses/trucks,
- clip with ambulance/police/fire truck if emergency model is installed.

For CPU testing, keep clips short:

```text
10 to 60 seconds, 480p or 720p, MP4/H.264 preferred
```

If a video fails decoding, convert it:

```bash
ffmpeg -i input_video.anything -vf scale=1280:-2 -c:v libx264 -preset fast -crf 23 -an test_traffic.mp4
```

## How The Camera Feed Works

### Server Camera / CCTV / RTSP

The server opens the source with OpenCV:

```python
cv2.VideoCapture(source)
```

Valid sources include:

- `0` for a webcam physically attached to the server machine,
- local video path like `static/uploads/sample.avi`,
- RTSP URL like `rtsp://user:pass@camera-ip:554/stream`,
- HTTP video stream if OpenCV supports it.

### Phone / Browser Camera

Browsers do not let a remote Flask server directly access a phone camera by webcam index. The correct user-friendly flow is:

1. Run the app on a public HTTPS URL or `localhost`.
2. Open that URL on the phone.
3. Log in.
4. Go to Live Monitoring.
5. Click Start Camera.

The browser captures frames with `getUserMedia`, sends JPEG frames to:

```text
POST /api/analyze-browser-frame
```

The backend analyzes each frame with YOLO and updates the live dashboard.

## Webserver Flow

Main server:

```text
app.py
```

Important routes:

- `/live` - live camera and upload UI.
- `/performance` - model install/status UI.
- `/api/live/start` - starts server-side camera/RTSP/file monitoring.
- `/api/live/stop` - stops live monitoring.
- `/api/live/status` - returns latest metrics.
- `/api/live/frame` - MJPEG stream of annotated frames.
- `/api/analyze-image` - analyzes uploaded image.
- `/api/analyze-video` - analyzes uploaded video.
- `/api/analyze-browser-frame` - analyzes phone/browser camera frame.
- `/api/summary` - dashboard summary data.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Default login:

```text
admin / admin123
```

Generate demo data:

```bash
python training/generate_sample_data.py --days 7
```

## Detailed Usage Workflow

### 1. Start The App

```bash
python app.py
```

Open `http://127.0.0.1:5000` and log in.

### 2. Check Models

Go to:

```text
Models
```

Confirm:

- Vehicle YOLO is Ready or using a reference fallback.
- Emergency YOLO is Ready only if `models/yolo/emergency/best.pt` exists.
- LSTM and XGBoost can show Baseline/Formula before training.

### 3. Test With Uploaded Video

Go to:

```text
Live -> Upload and Analyze -> Video Analysis
```

Choose any video file such as `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.wmv` or `.3gp`.

Recommended settings:

```text
Sample every: 10 to 20 frames
Max frames: 60 to 150 on CPU
```

The result modal shows:

- average congestion,
- average density,
- total CO2e,
- fuel waste,
- peak frame,
- left/right lane balance,
- vehicle type breakdown,
- analyzed preview frames.

### 4. Test With Server Camera Or Video Path

Go to:

```text
Live -> Server / RTSP / File
```

Use:

```text
0
```

only if the server machine has a webcam. For a local video path, use:

```text
static/uploads/cctv052x2004080516x01638.avi
```

For CCTV:

```text
rtsp://username:password@camera-ip:554/stream
```

### 5. Test With Phone Camera

For local laptop testing, use the laptop browser camera on `localhost`.

For phone testing:

```bash
python app.py
ngrok http 5000
```

Open the HTTPS ngrok URL on the phone, log in and click:

```text
Live -> Phone / Browser Camera -> Start Camera
```

### 6. Review Stored Data

Use:

- Dashboard for summary.
- Traffic page for density/congestion history.
- Emissions page for pollutant estimates.
- Alerts page for emergency detections.
- History page for database table views.
- Reports page for PDF/CSV export.

## Install YOLO .pt Models

From the web UI:

```text
Models -> Install Vehicle YOLO .pt
Models -> Install Emergency YOLO best.pt
```

From terminal for emergency:

```bash
python training/download_emergency_model.py --url "https://trusted-host/best.pt"
```

Vehicle model install through the UI copies to:

```text
models/yolo/traffic_density_best.pt
```

Emergency model install copies to:

```text
models/yolo/emergency/best.pt
```

## Docker

```bash
docker compose up --build
```

Open:

```text
http://127.0.0.1:5000
```

The compose file mounts:

- `database.db`,
- `models/`,
- `reports/generated/`.

That keeps model files and results outside the container image.

## Public URL / Phone Camera

For a temporary public URL during a demo, use a tunnel:

```bash
python app.py
ngrok http 5000
```

Then open the HTTPS ngrok URL on the phone. Browser camera APIs require HTTPS except on localhost.

For production, prefer a real container host:

- Render,
- Railway,
- Fly.io,
- Google Cloud Run,
- AWS/GCP/Azure VM.

## Vercel Integration Feasibility

Vercel is not a good host for this whole project because YOLO/OpenCV needs:

- large `.pt` files,
- long-running Python process,
- video upload endpoints,
- streaming MJPEG endpoint,
- CPU/GPU inference,
- writable storage/database.

Recommended architecture:

```text
Vercel static/frontend shell
        |
        v
Flask API on Render/Railway/Fly.io/Cloud Run/VM
        |
        v
PostgreSQL or SQLite for demo, object storage for uploads/models
```

For student demo use, running the Flask app directly on Render/Railway/Fly.io or with ngrok is simpler than splitting the frontend.

## Why This Is An AI Project

The project uses multiple AI/ML components:

- YOLO object detection for vehicles.
- YOLO transfer learning for emergency vehicles.
- Computer vision ROI/lane density analysis.
- LSTM congestion forecasting.
- XGBoost/factor-based emission prediction.
- Automated live video inference and reporting.

The system does not only display videos. It interprets frames, extracts structured traffic data, estimates pollution impact, detects emergency events and stores analytics for city traffic decision-making.

## Tools And Libraries

- Flask for webserver and routes.
- OpenCV for camera/video decoding and frame drawing.
- Ultralytics YOLO for `.pt` inference.
- NumPy for geometry and frame math.
- XGBoost for emission regression models.
- PyTorch through Ultralytics for YOLO/LSTM model support.
- SQLite for persistence.
- Bootstrap, Bootstrap Icons and Chart.js for the UI.
- ReportLab for PDF reports.
- Docker for containerized deployment.

## Limitations

- Real emission values require local vehicle age, fuel type, speed profiles, temperature, road grade and regulatory models such as MOVES/COPERT.
- Browser camera requires HTTPS or localhost.
- RTSP access depends on camera credentials and network reachability.
- CPU inference works but real-time multi-camera inference is much better on a GPU.
- Emergency detection requires a real emergency-trained `.pt`; the app intentionally does not simulate those detections.
