set(Examples
    basic
    matrix_clone
    float_type
    vector_begin
    normal_example
)
foreach(ex ${Examples})
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} ${OpenCV_LIBS})    
endforeach(ex)


