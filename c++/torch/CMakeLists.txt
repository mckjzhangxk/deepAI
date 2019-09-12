cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch)

find_package(Torch REQUIRED)

add_executable(helloTorch main.cpp Net.cpp Net.h)
target_link_libraries(helloTorch "${TORCH_LIBRARIES}")
set_property(TARGET helloTorch PROPERTY CXX_STANDARD 11)



