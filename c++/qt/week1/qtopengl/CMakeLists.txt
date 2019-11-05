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

add_executable(qtopengl
  main.cpp
  mainwindow.cpp
  mainwindow.h
  mainwindow.ui
  mesh.h
  mesh.cpp
  draw.h
  draw.cpp
  Camera.h
  Camera.cpp
  material.h
  material.cpp
  dialogmaterial.h
  dialogmaterial.cpp
  dialogmaterial.ui
)

#target_link_libraries(qtopengl PRIVATE Qt5::Widgets)
#qt5_use_modules(qtopengl Widgets OpenGL)
#target_link_libraries(qtopengl PRIVATE glut GL GLU)


qt5_use_modules(qtopengl Widgets OpenGL)
target_link_libraries(qtopengl ${QT_LIBRARIES} ${OPENGL_LIBRARIES})
target_link_libraries(qtopengl vecmath)

