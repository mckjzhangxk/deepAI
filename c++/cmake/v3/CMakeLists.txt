cmake_minimum_required (VERSION 2.6)
project (Tutorial)

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("include")

#add_subdirectory (lib)
#add library, name:path of library
add_library(MathFunctions mysqrt.cxx)
add_subdirectory(MathFunctions)


add_executable(Tutorial tutorial.cxx)
#target_link_libraries (Tutorial MathFunctions)
