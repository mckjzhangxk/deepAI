cmake_minimum_required(VERSION 3.0)
project(TEST_SHADE_LIB)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(hello SHARED hello.cpp)
add_executable(main main.cpp)
target_link_libraries(main hello)