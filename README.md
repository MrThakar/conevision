# ConeVision

A traffic cone detector that runs in real time on a laptop CPU. I trained a YOLOv8n model in PyTorch, exported it to ONNX, quantized it to INT8, and wrote a C++ detector around ONNX Runtime that works on images, video files, and a webcam.

I'm building this as the perception part of a small autonomous robot (ROS 2 + Gazebo), which is the next step.

<!-- Demo GIF goes here -->

## Results

Accuracy on the test split (54 images, 100 cones), which wasn't used during training:

| Model | mAP50 | Precision | Recall |
|---|---|---|---|
| FP32 | 0.846 | 0.848 | 0.727 |
| INT8 | 0.852 | 0.866 | 0.713 |

Speed per image on my MacBook Pro (M3 Pro, CPU only, ONNX Runtime 1.32). Each number is from 54 test images run 3 times after a warm-up:

| Pipeline | Median | p95 | FPS |
|---|---|---|---|
| PyTorch (Ultralytics) | 47.3 ms | 59.1 ms | 21 |
| Python + ONNX Runtime, FP32 | 31.0 ms | 37.3 ms | 32 |
| Python + ONNX Runtime, INT8 | 15.8 ms | 19.7 ms | 63 |
| C++ + ONNX Runtime, FP32 | 29.7 ms | 33.1 ms | 34 |
| C++ + ONNX Runtime, INT8 | 15.3 ms | 16.4 ms | 65 |

- INT8 is about 2x faster and 3.6x smaller than FP32, with the same accuracy.
- C++ and Python have similar typical speed, but C++ has fewer slow frames (p95 16.4 vs 19.7 ms).
- The C++ detector finds exactly the same boxes as the Python version.

## How it works

1. **Train:** fine-tune YOLOv8n on 650 cone images in Google Colab (about 9 minutes on a T4 GPU).
2. **Optimize:** export to ONNX and quantize to INT8.
3. **Deploy:** the C++ detector resizes each frame, runs the model with ONNX Runtime, and filters the results with its own non-maximum suppression (NMS).

I verified the pipeline in Python against Ultralytics first, then checked the C++ version against the Python one.

## Things that went wrong

- **The first INT8 model detected nothing.** YOLOv8's output mixes box coordinates (0–640) with scores (0–1), and one 8-bit range for both rounded every score to zero. Quantizing only the convolution layers fixed it.
- **C++ INT8 was 1.7x slower than Python with the same model.** The Homebrew build of ONNX Runtime lacks Arm's KleidiAI INT8 kernels, so the build now links Microsoft's official one.

## Limitations

Almost all of the training images are upright orange cones, so the model misses green or patterned cones and cones lying on their side. I plan to fix this with stronger colour augmentation and a more varied dataset.

## Running it

You'll need macOS on Apple Silicon, [Homebrew](https://brew.sh), and [uv](https://docs.astral.sh/uv/).

```bash
# Python environment: training tools, benchmarks, and the official ONNX Runtime
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# C++ dependencies and build
brew install cmake opencv onnxruntime
cmake -S cpp -B cpp/build
cmake --build cpp/build -j
```

The trained model isn't in the repo. Train it with [`training/train_cones.ipynb`](training/train_cones.ipynb) in Colab, put `conevision_yolov8n.onnx` in `models/`, then make the INT8 version:

```bash
.venv/bin/python training/quantize_int8.py
```

Run the detector:

```bash
./cpp/build/conevision models/conevision_yolov8n_int8.onnx photo.jpg   # saves photo_detected.jpg
./cpp/build/conevision models/conevision_yolov8n_int8.onnx video.mp4   # press q to quit
./cpp/build/conevision models/conevision_yolov8n_int8.onnx 0           # webcam
```

Benchmarks:

```bash
./cpp/build/benchmark models/conevision_yolov8n_int8.onnx data/traffic-cone/test/images
.venv/bin/python benchmarks/bench_python.py
```

## Next

- Detect non-orange and fallen cones (colour augmentation, more data)
- TensorRT backend on an NVIDIA GPU
- ROS 2 node that drives a simulated robot to the cones in Gazebo

## Dataset

[Traffic Cone dataset, version 3](https://universe.roboflow.com/project-k84yn/traffic-cone-ry08d/dataset/3), from Roboflow Universe, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
