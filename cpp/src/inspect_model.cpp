// Loads an ONNX model and prints the shape of its inputs and outputs.
// This is the "hello world" that proves the C++ toolchain works.
//
//   ./inspect_model ../../models/yolov8n.onnx

#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <string>
#include <vector>

static std::string shapeToString(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        out += (shape[i] < 0 ? "?" : std::to_string(shape[i]));
        if (i + 1 < shape.size()) out += ", ";
    }
    return out + "]";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
        return 1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "conevision");
    Ort::SessionOptions options;
    Ort::Session session(env, argv[1], options);
    Ort::AllocatorWithDefaultOptions allocator;

    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Input  " << name.get() << " " << shapeToString(shape) << "\n";
    }
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Output " << name.get() << " " << shapeToString(shape) << "\n";
    }
    return 0;
}
