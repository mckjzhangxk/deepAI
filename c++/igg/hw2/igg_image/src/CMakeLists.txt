add_subdirectory(igg_image)
include_directories(igg_image)
add_executable(initial_matrix main.cpp)
target_link_libraries(initial_matrix image)
