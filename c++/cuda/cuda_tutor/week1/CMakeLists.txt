project(week1)
cmake_minimum_required(VERSION 3.0)


find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(square square.cpp)
CUDA_ADD_EXECUTABLE(squareCuda square_cuda.cu)

target_link_libraries(squareCuda ${CUDA_LIBRARIES})
