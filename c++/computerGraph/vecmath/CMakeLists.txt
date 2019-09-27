cmake_minimum_required(VERSION 3.5)
project(MyProject)
file(GLOB header include/*.h)
file(GLOB source src/*.cpp)
add_library(
    vecmath
    ${header}
    ${source}
)