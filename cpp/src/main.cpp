// ConeVision demo: detects traffic cones in an image, a video file, or a webcam feed.
//
//   ./conevision <model.onnx> <image>         -> saves <image name>_detected.jpg in the current folder
//   ./conevision <model.onnx> <video file>    -> plays with boxes drawn
//   ./conevision <model.onnx> 0               -> webcam 0 (press q to quit)

#include "detector.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

void draw(cv::Mat& frame, const std::vector<conevision::Detection>& detections) {
    const cv::Scalar orange(0, 140, 255);  // BGR
    for (const auto& d : detections) {
        cv::rectangle(frame, d.box, orange, 2);
        char label[32];
        std::snprintf(label, sizeof(label), "cone %.2f", d.score);
        cv::putText(frame, label, cv::Point(int(d.box.x), std::max(int(d.box.y) - 6, 12)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, orange, 2);
    }
}

bool is_image(const std::string& path) {
    auto ext = std::filesystem::path(path).extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

int run_image(conevision::Detector& detector, const std::string& path) {
    cv::Mat image = cv::imread(path);
    if (image.empty()) {
        std::cerr << "Could not read image: " << path << "\n";
        return 1;
    }
    auto detections = detector.detect(image);
    draw(image, detections);

    // Saved in the current folder, never next to the input (keeps datasets clean).
    std::string out = std::filesystem::path(path).stem().string() + "_detected.jpg";
    cv::imwrite(out, image);

    const auto& t = detector.last_times();
    std::cout << detections.size() << " cone(s) found in " << t.total_ms() << " ms (pre "
              << t.preprocess_ms << ", inference " << t.inference_ms << ", post " << t.postprocess_ms
              << ")\nSaved " << out << "\n";
    return 0;
}

int run_stream(conevision::Detector& detector, const std::string& source) {
    const bool webcam = source.size() == 1 && std::isdigit(static_cast<unsigned char>(source[0]));
    cv::VideoCapture cap = webcam ? cv::VideoCapture(source[0] - '0') : cv::VideoCapture(source);
    if (!cap.isOpened()) {
        std::cerr << "Could not open " << (webcam ? "camera " : "video ") << source << "\n";
        return 1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        auto detections = detector.detect(frame);
        draw(frame, detections);

        const double ms = detector.last_times().total_ms();
        char fps[48];
        std::snprintf(fps, sizeof(fps), "%.1f ms  (%.0f FPS)", ms, 1000.0 / ms);
        cv::putText(frame, fps, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("ConeVision", frame);
        if ((cv::waitKey(1) & 0xFF) == 'q') break;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <image | video | camera index>\n";
        return 1;
    }
    conevision::Detector detector({argv[1]});
    const std::string source = argv[2];
    return is_image(source) ? run_image(detector, source) : run_stream(detector, source);
}
