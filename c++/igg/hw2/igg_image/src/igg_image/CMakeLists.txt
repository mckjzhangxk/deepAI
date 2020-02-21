#################### Import the 'io_tools' library here ########################
find_library(IOTOOLS io_tools
    PATH ${PROJECT_SOURCE_DIR}/lib
)
add_library(image image.cpp)
target_link_libraries(image ${IOTOOLS})
#################### Create a library 'image' below this line ##################
