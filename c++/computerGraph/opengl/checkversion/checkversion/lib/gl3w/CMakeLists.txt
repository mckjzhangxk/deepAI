cmake_minimum_required(VERSION 3.5)
project(gl3w)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
include_directories(../../include)
add_library(
    gl3w
    gl3w.c
)
