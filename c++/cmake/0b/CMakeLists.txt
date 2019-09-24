cmake_minimum_required(VERSION 3.15)
project(MyAI VERSION 1.0.0)
add_library(
    mylib
    hello.h
    hello.cpp
)
add_executable(MyAI main.cpp)
target_link_libraries(MyAI PRIVATE mylib)

#cmake -D BUILD_SHARED_LIBS=TRUE
