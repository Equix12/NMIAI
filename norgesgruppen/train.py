#!/usr/bin/env python3
"""
Train YOLOv8 on NorgesGruppen grocery shelf dataset.

Usage:
    source /home/haava/NMAI/.venv/bin/activate
    python train.py
"""

import torch
# Patch torch.load for ultralytics 8.1.0 compatibility with PyTorch 2.6
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).parent
DATASET_YAML = BASE / "dataset.yaml"

# Use YOLOv8m - good balance of speed/accuracy for GTX 1060 6GB
# Options: yolov8n (fast), yolov8s, yolov8m, yolov8l (slow but better)
model = YOLO("yolov8m.pt")

results = model.train(
    data=str(DATASET_YAML),
    epochs=80,
    imgsz=640,
    batch=4,           # GTX 1060 6GB - conservative batch size
    device=0,
    workers=4,
    patience=20,       # Early stopping
    save=True,
    project=str(BASE / "runs"),
    name="yolov8m_grocery",
    exist_ok=True,
    # Augmentation
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    # Optimizer
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
    # FP16 for speed on limited VRAM
    amp=True,
)

print(f"\nTraining complete!")
print(f"Best weights: {BASE / 'runs' / 'yolov8m_grocery' / 'weights' / 'best.pt'}")
