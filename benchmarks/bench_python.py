"""Python speed baseline for the ConeVision detector.

Times two ways of running the model on the test images (CPU only):
  A. Ultralytics PyTorch: model.predict(), the usual Python approach.
  B. ONNX Runtime with hand-written pre/post-processing: the same steps the
     C++ detector will do, so the two can be compared directly.

Usage (from the repo root):
  .venv/bin/python benchmarks/bench_python.py
"""

import glob
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
PT_MODEL = ROOT / "models" / "conevision_yolov8n.pt"
ONNX_MODEL = ROOT / "models" / "conevision_yolov8n.onnx"
INT8_MODEL = ROOT / "models" / "conevision_yolov8n_int8.onnx"
IMAGES = sorted(glob.glob(str(ROOT / "data" / "traffic-cone" / "test" / "images" / "*")))

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
WARMUP = 5
REPEATS = 3


def letterbox(img, size=INPUT_SIZE):
    """Resize keeping the aspect ratio, then pad to a square with grey."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(img):
    """BGR image -> float32 tensor [1, 3, 640, 640] scaled to 0..1."""
    canvas, scale, pad_x, pad_y = letterbox(img)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    return tensor, scale, pad_x, pad_y


def postprocess(output, scale, pad_x, pad_y):
    """Output [1, 5, 8400] -> list of (x, y, w, h, score) in original-image pixels."""
    preds = output[0].T                      # [8400, 5]: cx, cy, w, h, score
    preds = preds[preds[:, 4] >= CONF_THRESHOLD]
    if len(preds) == 0:
        return []
    cx, cy, w, h, scores = preds.T
    x = (cx - w / 2 - pad_x) / scale         # undo the letterbox padding and resize
    y = (cy - h / 2 - pad_y) / scale
    boxes = np.stack([x, y, w / scale, h / scale], axis=1)
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
    return [(*boxes[i], scores[i]) for i in np.array(keep).flatten()]


def summarize(name, times_ms):
    med = statistics.median(times_ms)
    print(f"{name:<34} median {med:7.2f} ms   mean {statistics.mean(times_ms):7.2f} ms   "
          f"({1000 / med:5.1f} FPS)")


def bench_pytorch(images):
    model = YOLO(str(PT_MODEL))
    for img in images[:WARMUP]:
        model.predict(img, device="cpu", conf=CONF_THRESHOLD, verbose=False)
    times = []
    for _ in range(REPEATS):
        for img in images:
            t0 = time.perf_counter()
            model.predict(img, device="cpu", conf=CONF_THRESHOLD, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
    summarize("A. PyTorch (Ultralytics), total", times)


def bench_onnxruntime(images, model_path, label):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def run(img):
        t0 = time.perf_counter()
        tensor, scale, pad_x, pad_y = preprocess(img)
        t1 = time.perf_counter()
        output = session.run(None, {input_name: tensor})[0]
        t2 = time.perf_counter()
        detections = postprocess(output, scale, pad_x, pad_y)
        t3 = time.perf_counter()
        return detections, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    for img in images[:WARMUP]:
        run(img)
    pre, inf, post, total = [], [], [], []
    for _ in range(REPEATS):
        for img in images:
            _, a, b, c = run(img)
            pre.append(a); inf.append(b); post.append(c); total.append(a + b + c)
    summarize(f"{label}, total", total)
    summarize("   - preprocess", pre)
    summarize("   - inference", inf)
    summarize("   - postprocess", post)


def main():
    images = [cv2.imread(p) for p in IMAGES]  # load once so disk reads aren't timed
    print(f"{len(images)} test images, {REPEATS} repeats, CPU only, ONNX Runtime {ort.__version__}\n")
    bench_pytorch(images)
    bench_onnxruntime(images, ONNX_MODEL, "B. ONNX Runtime FP32 (Python)")
    if INT8_MODEL.exists():
        bench_onnxruntime(images, INT8_MODEL, "C. ONNX Runtime INT8 (Python)")


if __name__ == "__main__":
    main()
