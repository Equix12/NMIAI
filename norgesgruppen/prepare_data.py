#!/usr/bin/env python3
"""
Convert COCO annotations to YOLO format and create train/val split.
"""

import json
import random
from pathlib import Path

random.seed(42)

BASE = Path(__file__).parent
DATASET = BASE / "dataset" / "train"
ANNOTATIONS = DATASET / "annotations.json"
IMAGES_DIR = DATASET / "images"
YOLO_DIR = BASE / "yolo_dataset"

# Output directories
for split in ["train", "val"]:
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# Load COCO annotations
with open(ANNOTATIONS) as f:
    coco = json.load(f)

print(f"Images: {len(coco['images'])}")
print(f"Annotations: {len(coco['annotations'])}")
print(f"Categories: {len(coco['categories'])}")

# Build lookup
images_by_id = {img["id"]: img for img in coco["images"]}
anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

# Train/val split (90/10)
image_ids = list(images_by_id.keys())
random.shuffle(image_ids)
split_idx = int(len(image_ids) * 0.9)
train_ids = set(image_ids[:split_idx])
val_ids = set(image_ids[split_idx:])
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

# Convert annotations to YOLO format
for img_id, img_info in images_by_id.items():
    split = "train" if img_id in train_ids else "val"
    img_w, img_h = img_info["width"], img_info["height"]
    filename = img_info["file_name"]

    # Symlink image
    src = IMAGES_DIR / filename
    dst = YOLO_DIR / "images" / split / filename
    if not dst.exists():
        dst.symlink_to(src.resolve())

    # Write YOLO label file
    label_file = YOLO_DIR / "labels" / split / (Path(filename).stem + ".txt")
    anns = anns_by_image.get(img_id, [])

    with open(label_file, "w") as f:
        for ann in anns:
            cat_id = ann["category_id"]
            # COCO bbox: [x, y, width, height] (absolute pixels)
            bx, by, bw, bh = ann["bbox"]
            # YOLO format: class x_center y_center width height (normalized)
            x_center = (bx + bw / 2) / img_w
            y_center = (by + bh / 2) / img_h
            w_norm = bw / img_w
            h_norm = bh / img_h
            # Clamp to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))
            f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

# Create dataset YAML
num_categories = len(coco["categories"])
cat_names = {c["id"]: c["name"] for c in coco["categories"]}

import yaml
yaml_data = {
    "path": str(YOLO_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": num_categories,
    "names": {i: cat_names.get(i, f"class_{i}") for i in range(num_categories)},
}
yaml_path = BASE / "dataset.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

print(f"\nYOLO dataset created at: {YOLO_DIR}")
print(f"Dataset YAML: {yaml_path}")
print(f"Categories: {num_categories}")

# Stats
total_anns = sum(len(anns_by_image.get(i, [])) for i in train_ids)
val_anns = sum(len(anns_by_image.get(i, [])) for i in val_ids)
print(f"Train annotations: {total_anns}")
print(f"Val annotations: {val_anns}")
