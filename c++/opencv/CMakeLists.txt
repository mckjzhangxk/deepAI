cmake_minimum_required(VERSION 2.8)
project( DISPLAY_IMAGE)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(displayimage begin.cc)
target_link_libraries(displayimage ${OpenCV_LIBS})

add_executable(readimage read.cc)
target_link_libraries(readimage ${OpenCV_LIBS})

add_executable(convertimage convert.cc)
target_link_libraries(convertimage ${OpenCV_LIBS})

add_executable(core core.cc)
target_link_libraries(core ${OpenCV_LIBS})
add_executable(matrix matrix.cc)
target_link_libraries(matrix ${OpenCV_LIBS})
