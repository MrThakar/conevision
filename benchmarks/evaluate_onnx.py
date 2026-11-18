"""Scores the FP32 and INT8 ONNX models on the held-out test set (mAP, precision, recall).

Usage (from the repo root):
  .venv/bin/python benchmarks/evaluate_onnx.py
"""

import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "traffic-cone"
MODELS = {
    "FP32": ROOT / "models" / "conevision_yolov8n.onnx",
    "INT8": ROOT / "models" / "conevision_yolov8n_int8.onnx",
}


def main():
    # The Roboflow data.yaml uses relative paths, so write one with absolute paths.
    config = {
        "path": str(DATASET),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["traffic_cone"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config, f)
        data_yaml = f.name

    results = {}
    for name, path in MODELS.items():
        m = YOLO(str(path), task="detect").val(
            data=data_yaml, split="test", imgsz=640, batch=1, device="cpu", verbose=False, plots=False
        )
        results[name] = m.box

    print(f"\n{'Model':<6} {'mAP50':>7} {'mAP50-95':>9} {'Precision':>10} {'Recall':>7}")
    for name, box in results.items():
        print(f"{name:<6} {box.map50:7.3f} {box.map:9.3f} {box.mp:10.3f} {box.mr:7.3f}")


if __name__ == "__main__":
    main()
