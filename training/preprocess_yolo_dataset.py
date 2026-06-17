"""Validate and summarise a YOLO-format emergency dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def validate_dataset(data_yaml: Path) -> dict:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = data_yaml.parent
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    expected = ["ambulance", "fire_truck", "police"]
    if [str(n) for n in names] != expected:
        raise ValueError(f"Expected names {expected}, got {names}")

    summary = {"classes": names, "splits": {}}
    for split in ("train", "val", "valid", "test"):
        rel = data.get(split)
        if not rel:
            continue
        image_dir = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if image_dir.name != "images":
            image_dir = image_dir / "images"
        label_dir = image_dir.parent / "labels"
        images = list(image_dir.glob("*.*")) if image_dir.exists() else []
        labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []
        summary["splits"][split] = {"images": len(images), "labels": len(labels)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO emergency dataset")
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    summary = validate_dataset(args.data)
    print(summary)


if __name__ == "__main__":
    main()
