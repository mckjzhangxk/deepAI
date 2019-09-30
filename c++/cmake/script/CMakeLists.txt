cmake_minimum_required(VERSION 3.5)
project(myproject VERSION 1.0.2)
message(STATUS "project name :${PROJECT_NAME}")
message(STATUS "project version : ${PROJECT_VERSION}")


#base stripe statement
# set(mychioce "b")
# if(mychioce STREQUAL "a")
#     message(STATUS "chioce A")
# elseif(mychioce STREQUAL "b")
#     message(STATUS "chioce B")
# endif("True")

#while
# set(myval 0)
# while(myval LESS 10)
#     message(STATUS "at step ${myval}")
#     math(EXPR myval "${myval}+1")
# endwhile()

#foreach
# foreach(i RANGE 1 11 2)
#     message(STATUS "${i}")
# endforeach()


function(myfunc ret)
    # foreach(ii IN LISTS ARGV)
    #     message(${ii})
    # endforeach()
    message(STATUS "parameter is ${ret}")
    set("retval" "SADSA" PARENT_SCOPE)
endfunction()

set(retval "hello")
set(retval "hello1")
myfunc(${retval})
message(STATUS "return value is ${retval}")

