#!/usr/bin/env python3
"""
Train YOLOv8l on NorgesGruppen grocery shelf dataset - v2.
Larger model, more epochs, tuned augmentation.
"""

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

model = YOLO("yolov8l.pt")

results = model.train(
    data=str(DATASET_YAML),
    epochs=150,
    imgsz=640,
    batch=2,            # YOLOv8l needs more VRAM, batch=2 for GTX 1060 6GB
    device=0,
    workers=4,
    patience=30,
    save=True,
    project=str(BASE / "runs"),
    name="yolov8l_grocery_v2",
    exist_ok=True,
    # Augmentation
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.15,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,         # Shelves are always upright
    degrees=5.0,
    translate=0.1,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    # Optimizer
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
    weight_decay=0.0005,
    amp=True,
    close_mosaic=20,
)

print(f"\nTraining complete!")
print(f"Best weights: {BASE / 'runs' / 'yolov8l_grocery_v2' / 'weights' / 'best.pt'}")
