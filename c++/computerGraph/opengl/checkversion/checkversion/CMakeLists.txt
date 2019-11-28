cmake_minimum_required(VERSION 3.5)

project(checkversion LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


set(CMAKE_CXX_STANDARD 11)
include_directories(include)
message("c compiler:${CMAKE_C_COMPILER}")
message("cxx compiler:${CMAKE_CXX_COMPILER}")


add_executable(checkversion main.cpp)
target_link_libraries(checkversion glfw3 glut GLEW GL dl)
target_link_libraries(checkversion X11 pthread)

