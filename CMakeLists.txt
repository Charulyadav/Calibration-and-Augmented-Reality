# Charul - 002535330
# Spring 2026
# CS 5330 Computer Vision
# Project 4: Calibration and Augmented Reality

cmake_minimum_required(VERSION 3.16)
project(Project4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(OpenCV_DIR "C:/Users/Charul/Downloads/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(detectCorners src/detectCorners.cpp)
target_link_libraries(detectCorners ${OpenCV_LIBS})

add_executable(detectPose src/detectPose.cpp)
target_link_libraries(detectPose ${OpenCV_LIBS})

add_executable(detectFeatures src/detectFeatures.cpp)
target_link_libraries(detectFeatures ${OpenCV_LIBS})