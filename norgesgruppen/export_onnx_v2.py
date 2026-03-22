#!/usr/bin/env python3
"""Export YOLOv8l to ONNX at imgsz=1280 for better detection on large shelf images."""

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).parent
WEIGHTS = BASE / "runs" / "yolov8l_grocery_v2" / "weights" / "best.pt"

model = YOLO(str(WEIGHTS))

# Export at 1280 - sandbox L4 has 24GB VRAM
model.export(format="onnx", imgsz=640, opset=17)

onnx_path = WEIGHTS.with_suffix(".onnx")
print(f"Exported to: {onnx_path}")
print(f"Size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
