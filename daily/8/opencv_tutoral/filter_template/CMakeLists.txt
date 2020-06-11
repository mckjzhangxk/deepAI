set(examples
    tm_search
)

foreach(ex ${examples})
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} ${OpenCV_LIBS}) 
endforeach(ex)

 
