cmake_minimum_required(VERSION 3.5)

project(layoutFirst LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_executable(layoutFirst
  main.cpp
  dialog.cpp
  dialog.h
  dialog.ui
)

target_link_libraries(layoutFirst PRIVATE Qt5::Widgets)
