"""Install a pretrained emergency YOLO `best.pt`.

There is no official Ultralytics emergency-vehicle checkpoint equivalent to
`yolov8n.pt`; emergency classes require a custom fine-tuned model. Use this
script with a trusted direct URL to a YOLOv8 `.pt` file trained for:

0 ambulance
1 fire_truck
2 police

Example:
  python training/download_emergency_model.py --url "https://.../best.pt"
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def install_model(url: str | None = None, source_file: str | None = None) -> Path:
    target = Path(config.EMERGENCY_MODEL)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".download")
    if source_file:
        src = Path(source_file)
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, tmp)
    elif url:
        if not url.startswith(("https://", "http://")):
            raise ValueError("Model URL must start with http:// or https://")
        urllib.request.urlretrieve(url, tmp)
    else:
        raise ValueError("Provide --url or --file")

    if tmp.stat().st_size < 1_000_000:
        tmp.unlink(missing_ok=True)
        raise ValueError("Downloaded file is too small to be a valid YOLO .pt checkpoint")
    tmp.replace(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Install emergency YOLO best.pt")
    parser.add_argument("--url", help="Trusted direct URL to best.pt")
    parser.add_argument("--file", help="Local best.pt file to copy")
    args = parser.parse_args()
    target = install_model(args.url, args.file)
    print(f"Installed emergency model: {target}")


if __name__ == "__main__":
    main()
