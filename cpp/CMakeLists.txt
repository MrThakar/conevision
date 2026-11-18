cmake_minimum_required(VERSION 3.20)
project(conevision CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(onnxruntime CONFIG REQUIRED)

# Reusable detector: preprocessing, ONNX Runtime inference, postprocessing + NMS.
add_library(conevision_detector src/detector.cpp)
target_include_directories(conevision_detector PUBLIC src)
target_link_libraries(conevision_detector PUBLIC onnxruntime::onnxruntime ${OpenCV_LIBS})
target_include_directories(conevision_detector PUBLIC ${OpenCV_INCLUDE_DIRS})

# Demo: image, video or webcam.
add_executable(conevision src/main.cpp)
target_link_libraries(conevision PRIVATE conevision_detector)

# Prints a model's input and output shapes.
add_executable(inspect_model src/inspect_model.cpp)
target_link_libraries(inspect_model PRIVATE onnxruntime::onnxruntime)
