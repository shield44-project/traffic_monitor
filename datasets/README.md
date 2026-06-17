# Datasets

This project is software-only and does not vendor large third-party datasets.
Place downloaded datasets under this directory, or keep them outside the repo
and point training scripts to their `data.yaml` files.

## Recommended Testing Video Sources

Included quick clips:

- `static/uploads/sample_video.mp4`
- `YOLOv8_Traffic_Density_Estimation/sample_video.mp4`
- `static/uploads/18437773-uhd_3840_2160_50fps.mp4`

Use these first from `Live -> Upload and Analyze -> Video Analysis`. For CPU,
start with `Sample every: 15` and `Max frames: 120`; use `Max frames: -1` only
when you intentionally want the whole clip analyzed.

1. AI City Challenge
   - Real traffic-camera videos and multi-camera city scenarios.
   - Good for testing live/video upload, congestion and lane density.
   - Website: `https://www.aicitychallenge.org/`
   - Current challenge page includes 2026 tasks and dataset access links.

2. UA-DETRAC
   - Fixed traffic-camera videos for vehicle detection and tracking.
   - Good for junction-style vehicle counting.
   - Search: `UA-DETRAC dataset official`.

3. BDD100K
   - Dashcam road-scene videos with varied weather, time and road types.
   - Good for general robustness tests.
   - Search: `BDD100K official download`.

4. CityFlow / AI City Challenge
   - Multi-camera city traffic benchmark.
   - Good for future multi-camera expansion.
   - Paper/search: `CityFlow A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification`.

5. BMD-45 CCTV Vehicle Detection
   - Recent large-scale CCTV traffic dataset direction for developing-city traffic.
   - Good for training/testing many vehicle categories in dense scenes.
   - Paper: `https://arxiv.org/abs/2604.24419`
   - Dataset card: `https://huggingface.co/datasets/iisc-aim/BMD-45`

6. Roboflow Top-View Vehicle Detection / Kaggle traffic datasets
   - YOLO-format image datasets for fine-tuning.
   - Good for creating a custom `best.pt`.

7. Pexels or other open-license traffic videos
   - Useful for short demo clips.
   - Confirm license before publishing or submitting results.

Good search terms for quick downloads:

```text
Pexels traffic street cars video
Pexels highway traffic video
Pixabay traffic road cars video
UA-DETRAC sample traffic video
BDD100K sample driving video
```

## Quick Test Video Guidance

For normal CPU testing, use short clips:

```text
10-60 seconds, 480p or 720p, MP4/H.264 preferred
```

If a file uploads but no frames decode, convert it:

```bash
ffmpeg -i input_video.anything -vf scale=1280:-2 -c:v libx264 -preset fast -crf 23 -an test_traffic.mp4
```

Then upload `test_traffic.mp4` from:

```text
Live -> Upload and Analyze -> Video Analysis
```

## Supported Dataset Types

1. General traffic videos for testing: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`,
   `.wmv`, `.3gp`, or anything OpenCV can decode.
2. YOLO-format vehicle datasets for training a vehicle/traffic `.pt`.
3. YOLO-format emergency datasets for ambulance, fire truck and police classes.
4. Historical CSV/synthetic data for traffic forecasting and emissions.

## Suggested Local Layout

```text
datasets/
  emergency/
  traffic_yolo/
  videos/
    low_density.mp4
    heavy_congestion.mp4
    night_traffic.mp4
  raw/
```

## Emergency YOLO Format

Expected layout:

```text
datasets/emergency/
  data.yaml
  train/images/*.jpg
  train/labels/*.txt
  valid/images/*.jpg
  valid/labels/*.txt
  test/images/*.jpg
  test/labels/*.txt
```

`data.yaml`:

```yaml
path: datasets/emergency
train: train/images
val: valid/images
test: test/images
names:
  0: ambulance
  1: fire_truck
  2: police
```

Each label file uses standard YOLO normalized coordinates:

```text
class_id x_center y_center width height
0 0.512 0.438 0.210 0.185
```

Validate before training:

```bash
python training/preprocess_yolo_dataset.py --data datasets/emergency/data.yaml
```

Fine-tune:

```bash
python training/train_emergency_yolo.py --data datasets/emergency/data.yaml
```

The best checkpoint is copied to:

```text
models/yolo/emergency/best.pt
```

## Vehicle YOLO Format

If training a vehicle detector, use standard YOLO layout:

```text
datasets/traffic_yolo/
  data.yaml
  train/images/*.jpg
  train/labels/*.txt
  valid/images/*.jpg
  valid/labels/*.txt
```

Example `data.yaml`:

```yaml
path: datasets/traffic_yolo
train: train/images
val: valid/images
names:
  0: car
  1: motorcycle
  2: bus
  3: truck
```

The app also supports models whose labels are:

```text
mobil, motor, truk, vehicle
```

Install the resulting `.pt` from:

```text
Models -> Install Vehicle YOLO .pt
```

## Literature/Report Dataset Notes

In a project report, mention:

- AI City Challenge for real city traffic videos.
- UA-DETRAC for fixed-camera vehicle detection/tracking.
- BDD100K for diverse driving-video scenes.
- CityFlow for multi-camera city traffic.
- BMD-45 for recent CCTV vehicle detection in dense developing-city traffic.
- Roboflow/Kaggle top-view vehicle datasets for YOLO-format fine-tuning.
- Custom emergency vehicle datasets for ambulance/fire truck/police detection.
