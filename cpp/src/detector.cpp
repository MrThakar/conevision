#include "detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace conevision {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = (a & b).area();
    const float uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

// Greedy non-maximum suppression: keep the most confident box, drop boxes that
// overlap it by more than `threshold`, repeat with the next box still standing.
std::vector<Detection> nms(std::vector<Detection> dets, float threshold) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!removed[j] && iou(dets[i].box, dets[j].box) > threshold) removed[j] = true;
        }
    }
    return kept;
}

}  // namespace

Detector::Detector(const DetectorConfig& config)
    : config_(config),
      env_(ORT_LOGGING_LEVEL_WARNING, "conevision"),
      session_(env_, config.model_path.c_str(), Ort::SessionOptions{}) {
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
    input_.resize(3 * config_.input_size * config_.input_size);
}

void Detector::preprocess(const cv::Mat& image) {
    const int size = config_.input_size;

    // Letterbox: resize keeping the aspect ratio, then centre on a grey square.
    scale_ = std::min(static_cast<float>(size) / image.rows, static_cast<float>(size) / image.cols);
    const int new_w = static_cast<int>(std::round(image.cols * scale_));
    const int new_h = static_cast<int>(std::round(image.rows * scale_));
    pad_x_ = (size - new_w) / 2;
    pad_y_ = (size - new_h) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x_, pad_y_, new_w, new_h)));

    // HWC (pixel by pixel) BGR bytes -> CHW (plane by plane) RGB floats in 0..1.
    // Each plane is a cv::Mat that writes straight into the model's input buffer.
    std::vector<cv::Mat> bgr;
    cv::split(canvas, bgr);
    const int plane = size * size;
    for (int c = 0; c < 3; ++c) {
        cv::Mat dst(size, size, CV_32F, input_.data() + c * plane);
        bgr[2 - c].convertTo(dst, CV_32F, 1.0 / 255.0);  // channel 0 = R = BGR index 2
    }
}

std::vector<Detection> Detector::postprocess(const float* output, int num_candidates) {
    // Output layout is [1, 5, N]: row 0 = centre x, 1 = centre y, 2 = width,
    // 3 = height, 4 = cone score. Candidate i's values are output[row * N + i].
    std::vector<Detection> candidates;
    for (int i = 0; i < num_candidates; ++i) {
        const float score = output[4 * num_candidates + i];
        if (score < config_.conf_threshold) continue;
        const float cx = output[0 * num_candidates + i];
        const float cy = output[1 * num_candidates + i];
        const float w = output[2 * num_candidates + i];
        const float h = output[3 * num_candidates + i];
        // Undo the letterbox: remove padding, then scale back to the original size.
        const float x = (cx - w / 2 - pad_x_) / scale_;
        const float y = (cy - h / 2 - pad_y_) / scale_;
        candidates.push_back({cv::Rect2f(x, y, w / scale_, h / scale_), score});
    }
    return nms(std::move(candidates), config_.nms_threshold);
}

std::vector<Detection> Detector::detect(const cv::Mat& image) {
    auto t0 = Clock::now();
    preprocess(image);
    times_.preprocess_ms = ms_since(t0);

    auto t1 = Clock::now();
    const std::array<int64_t, 4> shape{1, 3, config_.input_size, config_.input_size};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(memory, input_.data(), input_.size(),
                                                       shape.data(), shape.size());
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    times_.inference_ms = ms_since(t1);

    auto t2 = Clock::now();
    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();  // [1, 5, N]
    if (out_shape.size() != 3 || out_shape[1] != 5) {
        throw std::runtime_error("Expected a single-class YOLOv8 output of shape [1, 5, N]");
    }
    auto detections = postprocess(outputs[0].GetTensorData<float>(), static_cast<int>(out_shape[2]));
    times_.postprocess_ms = ms_since(t2);
    return detections;
}

}  // namespace conevision
