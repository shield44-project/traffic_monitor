# Deployment Guide

## Local Laptop

```bash
cd traffic_ai_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python training/generate_sample_data.py --days 3
python app.py
```

Open `http://127.0.0.1:5000` and sign in with `admin / admin123`.

## GPU Notes

`DEVICE=auto` uses CUDA when PyTorch detects an NVIDIA GPU. CPU fallback is
automatic. For a CPU-only install, install CPU PyTorch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Production Flask

```bash
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export DEBUG=false
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

Keep worker count low on student laptops because YOLO inference is memory
intensive.

## Docker

```bash
cd traffic_ai_system
docker compose up --build
```

For GPU containers, use an NVIDIA-enabled base image and run with the NVIDIA
container runtime. The default Dockerfile is CPU-oriented for portability.

## Vercel / Firebase / Public Hosting

Vercel and Firebase Hosting are useful for static frontends, but they are not a
good primary host for this full app because YOLO/OpenCV video processing needs:

- long-running Python processes,
- large model files,
- filesystem/database storage,
- CPU/GPU compute,
- streaming/video upload endpoints.

Recommended production layout:

```text
Browser / Firebase Hosting / Vercel frontend
        |
        v
Flask API on Cloud Run / Render / Railway / Fly.io / AWS ECS
        |
        v
SQLite for demo, PostgreSQL + object storage for production
```

Best options:

1. Google Cloud Run: easiest container deployment for Flask + OpenCV CPU
   inference. Use Cloud Storage for uploaded videos and model files.
2. Render/Railway/Fly.io: simple public URL for student demos. CPU inference is
   acceptable with YOLOv8n and sampled video frames.
3. AWS/GCP/Azure VM with NVIDIA GPU: best for real-time multi-camera inference.

If using Firebase or Vercel, host only the frontend there and set API calls to a
public Flask backend URL. Browser camera access also requires HTTPS in most
browsers, except `localhost`.

## Training Order

```bash
python training/generate_sample_data.py --days 14
python training/train_lstm.py
python training/train_xgboost.py
python training/train_emergency_yolo.py --data datasets/emergency/data.yaml
```

Generated artifacts:

```text
models/lstm/lstm_traffic.pt
models/lstm/scaler.json
models/xgboost/co2.json
models/xgboost/nox.json
models/xgboost/pm25.json
models/yolo/emergency/best.pt
data/*_metrics.json
```
