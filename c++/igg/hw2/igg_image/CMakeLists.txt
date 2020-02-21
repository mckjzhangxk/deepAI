################################################################################
####################### Do not modify this file. ###############################
################################################################################

project(initial_matrix)
cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O0")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)

################################################################################
####################### Do not modify this file. ###############################
################################################################################
