cmake_minimum_required(VERSION 3.20)
project(conevision CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(onnxruntime CONFIG REQUIRED)  # Homebrew: provides the headers

# Prefer Microsoft's official ONNX Runtime build from the project's Python environment.
# Homebrew's build lacks Arm's KleidiAI kernels, which made INT8 inference ~1.7x slower.
# Both are version 1.32.0, so Homebrew's headers match the official library.
file(GLOB OFFICIAL_ORT_LIB
     "${CMAKE_SOURCE_DIR}/../.venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.*.dylib")
if(OFFICIAL_ORT_LIB)
  list(GET OFFICIAL_ORT_LIB 0 OFFICIAL_ORT_LIB)
  get_target_property(ORT_INCLUDE onnxruntime::onnxruntime INTERFACE_INCLUDE_DIRECTORIES)
  add_library(ort_runtime SHARED IMPORTED)
  set_target_properties(ort_runtime PROPERTIES
    IMPORTED_LOCATION "${OFFICIAL_ORT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE}")
  # The binaries look for "libonnxruntime.1.dylib", but the wheel ships only the
  # fully versioned file, so link that name to it inside the build folder.
  set(ORT_LINK_DIR "${CMAKE_BINARY_DIR}/ort")
  file(MAKE_DIRECTORY "${ORT_LINK_DIR}")
  file(CREATE_LINK "${OFFICIAL_ORT_LIB}" "${ORT_LINK_DIR}/libonnxruntime.1.dylib" SYMBOLIC)
  set(CMAKE_BUILD_RPATH "${ORT_LINK_DIR}")
  message(STATUS "ONNX Runtime: official build ${OFFICIAL_ORT_LIB}")
else()
  add_library(ort_runtime INTERFACE IMPORTED)
  target_link_libraries(ort_runtime INTERFACE onnxruntime::onnxruntime)
  message(STATUS "ONNX Runtime: Homebrew build (run 'pip install onnxruntime' in .venv for faster INT8)")
endif()

# Reusable detector: preprocessing, ONNX Runtime inference, postprocessing + NMS.
add_library(conevision_detector src/detector.cpp)
target_include_directories(conevision_detector PUBLIC src)
target_link_libraries(conevision_detector PUBLIC ort_runtime ${OpenCV_LIBS})
target_include_directories(conevision_detector PUBLIC ${OpenCV_INCLUDE_DIRS})

# Demo: image, video or webcam.
add_executable(conevision src/main.cpp)
target_link_libraries(conevision PRIVATE conevision_detector)

# Speed benchmark on a folder of images (optionally dumps boxes to CSV).
add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE conevision_detector)

# Prints a model's input and output shapes.
add_executable(inspect_model src/inspect_model.cpp)
target_link_libraries(inspect_model PRIVATE ort_runtime)
