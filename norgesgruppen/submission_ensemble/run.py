import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion


IMGSZ = 640
CONF_THRESHOLD = 0.05
WBF_IOU_THR = 0.6
WBF_SKIP_THR = 0.05


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


def decode_yolo_output(output, conf_threshold, ratio, pad, orig_shape):
    predictions = output[0].T
    boxes_xywh = predictions[:, :4]
    scores = predictions[:, 4:]
    max_scores = scores.max(axis=1)
    mask = max_scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    max_scores = max_scores[mask]
    class_ids = scores.argmax(axis=1)

    if len(boxes_xywh) == 0:
        return np.array([]), np.array([]), np.array([])

    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad[0]) / ratio
    boxes_xyxy[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad[1]) / ratio
    boxes_xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - pad[0]) / ratio
    boxes_xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - pad[1]) / ratio

    orig_h, orig_w = orig_shape
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    return boxes_xyxy, max_scores, class_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    model_files = sorted(script_dir.glob("*.onnx"))
    print(f"Loading {len(model_files)} models: {[f.name for f in model_files]}")

    sessions = []
    for mf in model_files:
        sess = ort.InferenceSession(
            str(mf),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        sessions.append(sess)

    weights = [1.0] * len(sessions)
    predictions = []
    images = sorted(Path(args.input).iterdir())

    for img_path in images:
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        img = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = img.shape[:2]
        preprocessed, ratio, pad = letterbox(img, IMGSZ)

        blob = preprocessed.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        all_boxes = []
        all_scores = []
        all_labels = []

        for sess in sessions:
            input_name = sess.get_inputs()[0].name
            output = sess.run(None, {input_name: blob})[0]
            boxes, scores, class_ids = decode_yolo_output(
                output, CONF_THRESHOLD, ratio, pad, (orig_h, orig_w)
            )

            if len(boxes) == 0:
                all_boxes.append(np.array([]))
                all_scores.append(np.array([]))
                all_labels.append(np.array([]))
                continue

            norm_boxes = boxes.copy()
            norm_boxes[:, 0] /= orig_w
            norm_boxes[:, 1] /= orig_h
            norm_boxes[:, 2] /= orig_w
            norm_boxes[:, 3] /= orig_h
            norm_boxes = np.clip(norm_boxes, 0, 1)

            all_boxes.append(norm_boxes)
            all_scores.append(scores)
            all_labels.append(class_ids.astype(float))

        if all(len(b) == 0 for b in all_boxes):
            continue

        boxes_list = [b.tolist() if len(b) > 0 else [] for b in all_boxes]
        scores_list = [s.tolist() if len(s) > 0 else [] for s in all_scores]
        labels_list = [l.tolist() if len(l) > 0 else [] for l in all_labels]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights,
            iou_thr=WBF_IOU_THR,
            skip_box_thr=WBF_SKIP_THR,
        )

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1 = box[0] * orig_w
            y1 = box[1] * orig_h
            x2 = box[2] * orig_w
            y2 = box[3] * orig_h
            predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [
                    round(float(x1), 1),
                    round(float(y1), 1),
                    round(float(x2 - x1), 1),
                    round(float(y2 - y1), 1),
                ],
                "score": round(float(score), 3),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions for {len(images)} images")


if __name__ == "__main__":
    main()
