cmake_minimum_required(VERSION 3.5)

project(menu_dialog LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_executable(menu_dialog
  main.cpp
  mainwindow.cpp
  mainwindow.h
  mainwindow.ui
  dialog.h
  dialog.cpp
  dialog.ui
)

target_link_libraries(menu_dialog PRIVATE Qt5::Widgets vecmath)
