#!/usr/bin/env python3
"""Train YOLOv8x (largest) on NorgesGruppen grocery shelf dataset."""

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).parent
DATASET_YAML = BASE / "dataset.yaml"

model = YOLO("yolov8x.pt")

results = model.train(
    data=str(DATASET_YAML),
    epochs=150,
    imgsz=640,
    batch=1,            # YOLOv8x is huge, batch=1 for GTX 1060 6GB
    device=0,
    workers=4,
    patience=30,
    save=True,
    project=str(BASE / "runs"),
    name="yolov8x_grocery_v3",
    exist_ok=True,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.15,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,
    degrees=5.0,
    translate=0.1,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    optimizer="AdamW",
    lr0=0.0005,         # Lower LR for larger model
    lrf=0.01,
    warmup_epochs=5,
    weight_decay=0.0005,
    amp=True,
    close_mosaic=20,
)

print(f"\nTraining complete!")
print(f"Best weights: {BASE / 'runs' / 'yolov8x_grocery_v3' / 'weights' / 'best.pt'}")
