cmake_minimum_required(VERSION 3.0)
project(color)
set (CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


add_subdirectory(io_strategies)
# TODO: create image library from image.cpp.
# TODO: link image library to any other libraries if needed.
include_directories(io_strategies)
add_library(image image.cpp)
target_link_libraries(image strategies)
add_executable(main main.cpp)
target_link_libraries(main image)