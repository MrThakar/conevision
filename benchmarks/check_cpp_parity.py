"""Checks the C++ detector against the verified Python ONNX Runtime pipeline.

Run the C++ benchmark with a CSV path first, then pass the same model and CSV here:
  ./cpp/build/benchmark models/conevision_yolov8n.onnx data/traffic-cone/test/images outputs/cpp_fp32.csv
  .venv/bin/python benchmarks/check_cpp_parity.py models/conevision_yolov8n.onnx outputs/cpp_fp32.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_python import IMAGES, postprocess, preprocess  # noqa: E402
from check_parity import IOU_MATCH, iou  # noqa: E402


def main(model_path, csv_path):
    cpp = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cpp[row["image"]].append(tuple(float(row[k]) for k in ("x", "y", "w", "h", "score")))

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    total, matched, max_score_diff, differing = 0, 0, 0.0, []
    for path in IMAGES:
        name = Path(path).name
        tensor, scale, pad_x, pad_y = preprocess(cv2.imread(path))
        py_boxes = postprocess(session.run(None, {input_name: tensor})[0], scale, pad_x, pad_y)
        cpp_boxes = cpp.get(name, [])

        total += len(py_boxes)
        hits = 0
        for p in py_boxes:
            best = max(cpp_boxes, key=lambda c: iou(p[:4], c[:4]), default=None)
            if best is not None and iou(p[:4], best[:4]) >= IOU_MATCH:
                hits += 1
                max_score_diff = max(max_score_diff, abs(float(p[4]) - best[4]))
        matched += hits
        if hits != len(py_boxes) or len(cpp_boxes) != len(py_boxes):
            differing.append((name[:40], len(py_boxes), len(cpp_boxes)))

    print(f"Python boxes matched by C++: {matched}/{total}")
    print(f"Largest confidence difference on matched boxes: {max_score_diff:.4f}")
    print(f"Images that differ: {len(differing)}/{len(IMAGES)}")
    for name, n_py, n_cpp in differing:
        print(f"  {name}: python {n_py}, c++ {n_cpp}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
