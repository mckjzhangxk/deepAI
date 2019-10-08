cmake_minimum_required(VERSION 3.12)
project(three)
#添加 库RK4的目录
link_directories(.)


file(GLOB srcfile *.cpp)
add_executable(a3 ${srcfile})
target_link_libraries(a3 glut GL GLU vecmath RK4)