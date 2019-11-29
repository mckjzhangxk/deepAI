cmake_minimum_required(VERSION 3.5)

project(begin)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
link_directories(lib)

#add_executable(begin main.cpp myutil.cpp youtubeDemo.cpp)
add_executable(begin youtubeDemo.cpp myutil.cpp)
target_link_libraries(begin glfw3 GL GLEW gl3w dl pthread)
target_link_libraries(begin X11 Xrandr Xinerama Xi Xxf86vm Xcursor)
