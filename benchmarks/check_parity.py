"""Checks that the hand-written ONNX Runtime pipeline finds the same cones as Ultralytics.

For each test image, every Ultralytics box must have a matching ONNX Runtime box
(IoU >= 0.9). Small differences are expected from resize rounding.

Usage (from the repo root):
  .venv/bin/python benchmarks/check_parity.py
"""

import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO

from bench_python import IMAGES, ONNX_MODEL, PT_MODEL, CONF_THRESHOLD, preprocess, postprocess

IOU_MATCH = 0.9


def iou(a, b):
    """IoU of two boxes given as (x, y, w, h)."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    model = YOLO(str(PT_MODEL))
    session = ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    total_ref, matched, mismatched_images = 0, 0, []
    for path in IMAGES:
        img = cv2.imread(path)

        ref = model.predict(img, device="cpu", conf=CONF_THRESHOLD, verbose=False)[0].boxes
        ref_boxes = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in ref.xyxy.numpy()]

        tensor, scale, pad_x, pad_y = preprocess(img)
        ours = postprocess(session.run(None, {input_name: tensor})[0], scale, pad_x, pad_y)

        total_ref += len(ref_boxes)
        hits = sum(any(iou(r, o[:4]) >= IOU_MATCH for o in ours) for r in ref_boxes)
        matched += hits
        if hits != len(ref_boxes) or len(ours) != len(ref_boxes):
            mismatched_images.append((path.split("/")[-1][:40], len(ref_boxes), len(ours)))

    print(f"Reference boxes matched: {matched}/{total_ref}")
    print(f"Images with a different box count or unmatched box: {len(mismatched_images)}/{len(IMAGES)}")
    for name, n_ref, n_ours in mismatched_images:
        print(f"  {name}: ultralytics {n_ref} boxes, onnxruntime {n_ours} boxes")


if __name__ == "__main__":
    main()
