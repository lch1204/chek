cmake_minimum_required(VERSION 3.0.0)
project(chek VERSION 0.1.0 LANGUAGES CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(chek main.cpp)
target_link_libraries(chek ${OpenCV_LIBS})
