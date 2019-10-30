cmake_minimum_required(VERSION 3.5)

project(basicApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)
add_executable(basicApp main.cpp)
target_link_libraries(basicApp PRIVATE Qt5::Widgets Qt5::Core)
