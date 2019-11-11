cmake_minimum_required(VERSION 3.12)
project(three)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


file(GLOB srcfile *.cpp)
add_executable(a5 ${srcfile})
target_link_libraries(a5 glut GL GLU vecmath)