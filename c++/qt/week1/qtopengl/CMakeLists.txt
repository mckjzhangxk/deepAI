cmake_minimum_required(VERSION 3.5)

project(qtopengl LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

# Qt library
find_package(Qt5Widgets)
find_package(Qt5OpenGL)



# OpenGL & GLEW library
find_package(OpenGL)

file(GLOB header include/*)
file(GLOB src src/*/*)

include_directories(${header})
add_executable(qtopengl
  main.cpp
#  ${header}
  ${src}
 )



qt5_use_modules(qtopengl Widgets OpenGL)
target_link_libraries(qtopengl ${QT_LIBRARIES} ${OPENGL_LIBRARIES})
target_link_libraries(qtopengl vecmath)
target_include_directories(qtopengl PUBLIC ${header})
