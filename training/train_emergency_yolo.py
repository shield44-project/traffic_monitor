"""Fine-tune YOLOv8 for emergency vehicle detection.

This script uses pretrained Ultralytics weights and transfer learning. It does
not train object detection from scratch. Prepare a YOLO-format dataset with:

dataset/
  data.yaml
  train/images, train/labels
  valid/images, valid/labels

data.yaml names must be: ambulance, fire_truck, police

Example:
  python training/train_emergency_yolo.py --data datasets/emergency/data.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.device import resolve_device  # noqa: E402
from core.logger import get_logger  # noqa: E402

log = get_logger("train_emergency_yolo")


def train(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    device = resolve_device()
    model = YOLO(args.base_model)

    log.info("Stage 1: frozen transfer learning from %s", args.base_model)
    model.train(
        data=args.data,
        epochs=args.freeze_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(config.YOLO_DIR / "runs"),
        name="emergency_stage1",
        pretrained=True,
        freeze=args.freeze_layers,
        patience=10,
        workers=args.workers,
    )

    log.info("Stage 2: fine-tuning detection layers")
    results = model.train(
        data=args.data,
        epochs=args.finetune_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(config.YOLO_DIR / "runs"),
        name="emergency_finetune",
        pretrained=True,
        freeze=0,
        patience=15,
        workers=args.workers,
    )

    metrics = model.val(data=args.data, imgsz=args.imgsz, device=device)
    out_dir = config.YOLO_DIR / "emergency"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_source = Path(model.trainer.best) if getattr(model, "trainer", None) else None
    if best_source and best_source.exists():
        target = out_dir / "best.pt"
        target.write_bytes(best_source.read_bytes())
        log.info("Saved best checkpoint to %s", target)

    metric_payload = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "base_model": args.base_model,
        "data": args.data,
    }
    (config.DATA_DIR / "emergency_yolo_metrics.json").write_text(
        json.dumps(metric_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(metric_payload, indent=2))
    return metric_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 emergency detector")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Pretrained YOLOv8 weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--freeze-layers", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
