cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime-linux-x64-1.20.0")

project(object_tracking)

find_package(OpenCV 4.9 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include/googletest)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include/pugixml)

# Executable for create matrix exercise
add_executable (object_tracking src/main.cpp src/inferenceDriverONNX.cpp src/helpers.cpp src/trackManager.cpp)
target_link_libraries (object_tracking ${OpenCV_LIBRARIES})
target_link_libraries (object_tracking ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)

# Add the testing executable
add_executable(test tests/utest_object_tracking.cpp)
target_link_libraries(test gtest_main)