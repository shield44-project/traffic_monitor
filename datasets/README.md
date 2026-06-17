# Datasets

This project is software-only and does not include large third-party datasets in
the repository. Place datasets under this directory.

## Supported Sources

1. UA-DETRAC for general traffic videos and vehicle counts.
2. AI City Challenge data for multi-camera traffic analysis.
3. Roboflow emergency vehicle datasets for ambulance, fire truck and police
   vehicle detection.
4. Any custom YOLO-format dataset.

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
