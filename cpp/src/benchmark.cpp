// Times the C++ detector on a folder of images, matching benchmarks/bench_python.py:
// images are loaded up front, 5 warm-up runs, then every image is timed REPEATS times.
//
//   ./benchmark <model.onnx> <images folder> [boxes.csv]
//
// If a CSV path is given, every detection from the first timed pass is written to it
// (image,x,y,w,h,score) so benchmarks/check_cpp_parity.py can compare it with Python.

#include "detector.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kWarmup = 5;
constexpr int kRepeats = 3;

struct Stats {
    double median, mean, p95;
};

Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    const double median = n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    const double p95 = v[std::min(n - 1, static_cast<size_t>(0.95 * (n - 1) + 0.5))];
    return {median, mean, p95};
}

void print_row(const char* name, const std::vector<double>& times) {
    const Stats s = stats(times);
    std::printf("%-22s median %7.2f ms   mean %7.2f ms   p95 %7.2f ms   (%5.1f FPS)\n",
                name, s.median, s.mean, s.p95, 1000.0 / s.median);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <images folder> [boxes.csv]\n";
        return 1;
    }

    // Load every image first so disk reads are not part of the timing.
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(argv[2])) {
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    std::vector<cv::Mat> images;
    for (const auto& p : paths) images.push_back(cv::imread(p.string()));
    if (images.empty()) {
        std::cerr << "No images found in " << argv[2] << "\n";
        return 1;
    }

    conevision::Detector detector({argv[1]});
    for (int i = 0; i < kWarmup; ++i) detector.detect(images[i % images.size()]);

    std::ofstream csv;
    if (argc > 3) {
        csv.open(argv[3]);
        csv << "image,x,y,w,h,score\n";
    }

    std::vector<double> pre, inf, post, total;
    size_t boxes = 0;
    for (int r = 0; r < kRepeats; ++r) {
        for (size_t i = 0; i < images.size(); ++i) {
            const auto detections = detector.detect(images[i]);
            const auto& t = detector.last_times();
            pre.push_back(t.preprocess_ms);
            inf.push_back(t.inference_ms);
            post.push_back(t.postprocess_ms);
            total.push_back(t.total_ms());
            if (r == 0) {
                boxes += detections.size();
                if (csv.is_open()) {
                    for (const auto& d : detections) {
                        csv << paths[i].filename().string() << ',' << d.box.x << ',' << d.box.y << ','
                            << d.box.width << ',' << d.box.height << ',' << d.score << '\n';
                    }
                }
            }
        }
    }

    std::printf("%zu images, %d repeats, CPU, model %s\n", images.size(), kRepeats,
                fs::path(argv[1]).filename().string().c_str());
    std::printf("%zu cones detected (first pass)\n\n", boxes);
    print_row("C++ total", total);
    print_row("   - preprocess", pre);
    print_row("   - inference", inf);
    print_row("   - postprocess", post);
    return 0;
}
