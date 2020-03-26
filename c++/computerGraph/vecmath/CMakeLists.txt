cmake_minimum_required(VERSION 3.5)
project(MyProject)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
file(GLOB header include/*.h)
file(GLOB source src/*.cpp)
add_library(
    vecmath SHARED
    ${header}
    ${source}
)
add_executable(main main.cpp)
target_link_libraries(main vecmath)
