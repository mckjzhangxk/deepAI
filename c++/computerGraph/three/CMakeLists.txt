cmake_minimum_required(VERSION 3.12)
project(three)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#添加 库RK4的目录
link_directories(.)


file(GLOB srcfile *.cpp)
add_executable(a3 ${srcfile})
target_link_libraries(a3 glut GL GLU vecmath RK4)