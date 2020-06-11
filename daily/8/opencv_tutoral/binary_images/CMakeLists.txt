set(examples
    hello
    binary_example
    dilate_erode_example
    connectComponent_example
    dist_example
)

foreach(ex ${examples})
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} ${OpenCV_LIBS}) 
endforeach(ex)

 
