#!/usr/bin/env python3
"""Test different conf/NMS thresholds against val set to find optimal settings."""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort

BASE = Path(__file__).parent
VAL_IMAGES = BASE / "yolo_dataset" / "images" / "val"
ANNOTATIONS = BASE / "dataset" / "train" / "annotations.json"
MODEL = BASE / "runs" / "yolov8m_grocery" / "weights" / "best.onnx"  # use v1 for speed

# Load ground truth
with open(ANNOTATIONS) as f:
    coco = json.load(f)

images_by_id = {img["id"]: img for img in coco["images"]}
gt_by_image = {}
for ann in coco["annotations"]:
    gt_by_image.setdefault(ann["image_id"], []).append(ann)

# Get val image IDs
val_image_ids = set()
for img_path in VAL_IMAGES.iterdir():
    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        val_image_ids.add(int(img_path.stem.split("_")[-1]))

print(f"Val images: {len(val_image_ids)}")

# Load model
session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_h, new_w = int(h * r), int(w * r)
    resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    pad_h, pad_w = new_shape - new_h, new_shape - new_w
    top, left = pad_h // 2, pad_w // 2
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, r, (left, top)


def nms(boxes, scores, iou_threshold):
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def detect(output, conf_thresh, iou_thresh, ratio, pad, orig_shape):
    predictions = output[0].T
    boxes_xywh = predictions[:, :4]
    scores = predictions[:, 4:]
    max_scores = scores.max(axis=1)
    mask = max_scores > conf_thresh
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    max_scores = max_scores[mask]
    class_ids = scores.argmax(axis=1)
    if len(boxes_xywh) == 0:
        return [], [], []

    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad[0]) / ratio
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad[1]) / ratio
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad[0]) / ratio
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad[1]) / ratio

    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_shape[1])
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_shape[0])
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_shape[1])
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_shape[0])

    keep = nms(boxes_xyxy, max_scores, iou_thresh)
    return boxes_xyxy[keep], max_scores[keep], class_ids[keep]


def compute_iou(box1, box2):
    """box format: [x, y, w, h] COCO format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    return inter / (area1 + area2 - inter + 1e-6)


def evaluate(preds_by_image, gt_by_image, val_ids, iou_thresh=0.5):
    """Simple mAP-like evaluation: detection recall and precision."""
    total_gt = 0
    total_det = 0
    det_tp = 0
    cls_tp = 0

    for img_id in val_ids:
        gt_anns = gt_by_image.get(img_id, [])
        preds = preds_by_image.get(img_id, [])
        total_gt += len(gt_anns)
        total_det += len(preds)

        matched_gt = set()
        for pred in sorted(preds, key=lambda p: p["score"], reverse=True):
            best_iou = 0
            best_idx = -1
            for gi, gt in enumerate(gt_anns):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_iou >= iou_thresh and best_idx >= 0:
                matched_gt.add(best_idx)
                det_tp += 1
                if pred["category_id"] == gt_anns[best_idx]["category_id"]:
                    cls_tp += 1

    det_precision = det_tp / total_det if total_det > 0 else 0
    det_recall = det_tp / total_gt if total_gt > 0 else 0
    cls_precision = cls_tp / total_det if total_det > 0 else 0
    cls_recall = cls_tp / total_gt if total_gt > 0 else 0

    return {
        "gt": total_gt, "det": total_det,
        "det_tp": det_tp, "det_prec": det_precision, "det_recall": det_recall,
        "cls_tp": cls_tp, "cls_prec": cls_precision, "cls_recall": cls_recall,
    }


# Run inference once, cache raw outputs
print("Running inference on val set...")
raw_outputs = {}
for img_path in sorted(VAL_IMAGES.iterdir()):
    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue
    img_id = int(img_path.stem.split("_")[-1])
    img = np.array(Image.open(img_path).convert("RGB"))
    orig_h, orig_w = img.shape[:2]
    preprocessed, ratio, pad = letterbox(img, 640)
    blob = preprocessed.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    output = session.run(None, {input_name: blob})[0]
    raw_outputs[img_id] = (output, ratio, pad, (orig_h, orig_w))
    print(f"  {img_path.name}", end="", flush=True)
print(f"\n\nCached {len(raw_outputs)} images\n")

# Test different thresholds
print(f"{'conf':>6} {'iou':>6} | {'det':>5} {'det_P':>6} {'det_R':>6} | {'cls_P':>6} {'cls_R':>6} | {'score_est':>9}")
print("-" * 75)

for conf_thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
    for iou_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_by_image = {}
        for img_id, (output, ratio, pad, orig_shape) in raw_outputs.items():
            boxes, scores, class_ids = detect(output, conf_thresh, iou_thresh, ratio, pad, orig_shape)
            preds = []
            for box, score, cls_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box
                preds.append({
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                    "category_id": int(cls_id),
                })
            preds_by_image[img_id] = preds

        r = evaluate(preds_by_image, gt_by_image, val_image_ids)
        # Rough score estimate: 0.7 * det_recall + 0.3 * cls_recall
        score_est = 0.7 * r["det_recall"] + 0.3 * r["cls_recall"]
        print(f"{conf_thresh:>6.2f} {iou_thresh:>6.2f} | {r['det']:>5} {r['det_prec']:>6.3f} {r['det_recall']:>6.3f} | {r['cls_prec']:>6.3f} {r['cls_recall']:>6.3f} | {score_est:>9.4f}")
