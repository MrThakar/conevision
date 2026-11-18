"""Static INT8 quantization of the ConeVision ONNX model with ONNX Runtime.

Calibrates on training images (never test images) so the test score stays fair.

Usage (from the repo root):
  .venv/bin/python training/quantize_int8.py
"""

import glob
import random
import sys
from pathlib import Path

import cv2
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_python import preprocess  # same preprocessing as inference  # noqa: E402

FP32_MODEL = ROOT / "models" / "conevision_yolov8n.onnx"
PREPPED_MODEL = ROOT / "models" / "conevision_yolov8n_prepped.onnx"
INT8_MODEL = ROOT / "models" / "conevision_yolov8n_int8.onnx"
CALIBRATION_IMAGES = 100


class ConeCalibrationReader(CalibrationDataReader):
    """Feeds preprocessed training images to the calibrator one at a time."""

    def __init__(self, input_name):
        paths = sorted(glob.glob(str(ROOT / "data" / "traffic-cone" / "train" / "images" / "*")))
        random.Random(0).shuffle(paths)
        self.input_name = input_name
        self.paths = iter(paths[:CALIBRATION_IMAGES])

    def get_next(self):
        path = next(self.paths, None)
        if path is None:
            return None
        tensor, *_ = preprocess(cv2.imread(path))
        return {self.input_name: tensor}


def main():
    # Adds shape information and simplifies the graph, which quantization needs.
    quant_pre_process(str(FP32_MODEL), str(PREPPED_MODEL))

    quantize_static(
        model_input=str(PREPPED_MODEL),
        model_output=str(INT8_MODEL),
        calibration_data_reader=ConeCalibrationReader("images"),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        # Only quantize convolutions. The output head concatenates box pixels (0-640)
        # with class scores (0-1); squeezing both into one INT8 range zeroes the scores.
        op_types_to_quantize=["Conv"],
    )
    PREPPED_MODEL.unlink()

    fp32_mb = FP32_MODEL.stat().st_size / 1e6
    int8_mb = INT8_MODEL.stat().st_size / 1e6
    print(f"FP32: {fp32_mb:.1f} MB -> INT8: {int8_mb:.1f} MB ({fp32_mb / int8_mb:.1f}x smaller)")


if __name__ == "__main__":
    main()
