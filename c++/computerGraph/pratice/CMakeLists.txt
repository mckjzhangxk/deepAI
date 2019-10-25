cmake_minimum_required(VERSION 3.12)
project(myopencl)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB srcfile *.cpp)
add_executable(myzxk ${srcfile})
target_link_libraries(myzxk glut GL GLU vecmath  fltk_gl fltk dl m Xext X11)