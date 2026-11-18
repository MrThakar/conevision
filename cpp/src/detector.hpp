#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace conevision {

// One detected cone, in original-image pixel coordinates.
struct Detection {
    cv::Rect2f box;  // x, y (top-left), width, height
    float score;     // confidence, 0..1
};

// Time spent in each stage of the last detect() call, in milliseconds.
struct StageTimes {
    double preprocess_ms = 0;
    double inference_ms = 0;
    double postprocess_ms = 0;
    double total_ms() const { return preprocess_ms + inference_ms + postprocess_ms; }
};

struct DetectorConfig {
    std::string model_path;
    int input_size = 640;
    float conf_threshold = 0.25f;  // keep boxes at least this confident
    float nms_threshold = 0.45f;   // overlap above which duplicate boxes are removed
};

// Runs a single-class YOLOv8 ONNX model (output [1, 5, 8400]) with ONNX Runtime.
class Detector {
public:
    explicit Detector(const DetectorConfig& config);

    // Finds cones in a BGR image (as loaded by OpenCV).
    std::vector<Detection> detect(const cv::Mat& image);

    const StageTimes& last_times() const { return times_; }

private:
    // Letterbox + BGR->RGB + HWC->CHW + scale to 0..1, written into input_.
    void preprocess(const cv::Mat& image);
    std::vector<Detection> postprocess(const float* output, int num_candidates);

    DetectorConfig config_;
    Ort::Env env_;
    Ort::Session session_;
    std::string input_name_;
    std::string output_name_;

    std::vector<float> input_;  // reused buffer, 1 x 3 x 640 x 640
    float scale_ = 1.0f;        // letterbox resize factor
    int pad_x_ = 0, pad_y_ = 0; // letterbox padding
    StageTimes times_;
};

}  // namespace conevision
