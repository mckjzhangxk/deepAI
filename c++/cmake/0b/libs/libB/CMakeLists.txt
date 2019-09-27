add_library(
    myB
    include/B.h
    src/B.cpp
)
#表示 要链接myA这个库，必须包含这个头文件
target_include_directories(myB PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)