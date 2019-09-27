cmake_minimum_required(VERSION 3.12)
project(MyZXK)

add_subdirectory(libs)

add_executable(myzxk main.cpp)
target_link_libraries(myzxk myA)
target_link_libraries(myzxk myB)
set("myvar" "g++")
message(STATUS 完成${myvar})