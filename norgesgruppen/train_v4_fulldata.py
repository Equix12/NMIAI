#!/usr/bin/env python3
"""
Train YOLOv8l on ALL 248 images (no val holdout).
Also train YOLOv8m for ensemble diversity.
"""

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import json
import random
from pathlib import Path
from ultralytics import YOLO
import yaml

random.seed(42)

BASE = Path(__file__).parent
DATASET = BASE / "dataset" / "train"
ANNOTATIONS = DATASET / "annotations.json"
IMAGES_DIR = DATASET / "images"
YOLO_DIR = BASE / "yolo_fulldata"

# Create dirs
for split in ["train", "val"]:
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# Load annotations
with open(ANNOTATIONS) as f:
    coco = json.load(f)

images_by_id = {img["id"]: img for img in coco["images"]}
anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

# ALL images go to train, use a small subset as val too (for metrics only)
all_ids = list(images_by_id.keys())
random.shuffle(all_ids)
val_ids = set(all_ids[:10])  # 10 images for val metrics, but they're also in train

for img_id, img_info in images_by_id.items():
    img_w, img_h = img_info["width"], img_info["height"]
    filename = img_info["file_name"]

    # All images go to train
    src = IMAGES_DIR / filename
    train_dst = YOLO_DIR / "images" / "train" / filename
    if not train_dst.exists():
        train_dst.symlink_to(src.resolve())
    label_file = YOLO_DIR / "labels" / "train" / (Path(filename).stem + ".txt")
    anns = anns_by_image.get(img_id, [])
    with open(label_file, "w") as f:
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]
            x_c = max(0, min(1, (bx + bw / 2) / img_w))
            y_c = max(0, min(1, (by + bh / 2) / img_h))
            w_n = max(0, min(1, bw / img_w))
            h_n = max(0, min(1, bh / img_h))
            f.write(f"{ann['category_id']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

    # Subset also goes to val for metrics
    if img_id in val_ids:
        val_dst = YOLO_DIR / "images" / "val" / filename
        if not val_dst.exists():
            val_dst.symlink_to(src.resolve())
        val_label = YOLO_DIR / "labels" / "val" / (Path(filename).stem + ".txt")
        if not val_label.exists():
            val_label.symlink_to(label_file.resolve())

# Dataset YAML
num_cat = len(coco["categories"])
cat_names = {c["id"]: c["name"] for c in coco["categories"]}
yaml_data = {
    "path": str(YOLO_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": num_cat,
    "names": {i: cat_names.get(i, f"class_{i}") for i in range(num_cat)},
}
yaml_path = BASE / "dataset_full.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

print(f"Full dataset: {len(all_ids)} train, {len(val_ids)} val (subset of train)")
print(f"Total annotations: {sum(len(v) for v in anns_by_image.values())}")

# Train YOLOv8l on full data
import sys
model_size = sys.argv[1] if len(sys.argv) > 1 else "l"
model_name = f"yolov8{model_size}"
print(f"\nTraining {model_name} on full dataset...")

model = YOLO(f"{model_name}.pt")
results = model.train(
    data=str(yaml_path),
    epochs=100,
    imgsz=640,
    batch=2 if model_size in ("l", "x") else 4,
    device=0,
    workers=4,
    patience=25,
    save=True,
    project=str(BASE / "runs"),
    name=f"{model_name}_fulldata",
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
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
    weight_decay=0.0005,
    amp=True,
    close_mosaic=15,
)

print(f"\nDone! Best: {BASE / 'runs' / f'{model_name}_fulldata' / 'weights' / 'best.pt'}")
