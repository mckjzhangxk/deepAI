# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake/bin/cmake

# The command to remove a file.
RM = /opt/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhangxk/projects/deepAI/c++/cmake/0b

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhangxk/projects/deepAI/c++/cmake/0b/build

# Include any dependencies generated for this target.
include libs/libA/CMakeFiles/myA.dir/depend.make

# Include the progress variables for this target.
include libs/libA/CMakeFiles/myA.dir/progress.make

# Include the compile flags for this target's objects.
include libs/libA/CMakeFiles/myA.dir/flags.make

libs/libA/CMakeFiles/myA.dir/src/A.cpp.o: libs/libA/CMakeFiles/myA.dir/flags.make
libs/libA/CMakeFiles/myA.dir/src/A.cpp.o: ../libs/libA/src/A.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhangxk/projects/deepAI/c++/cmake/0b/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libs/libA/CMakeFiles/myA.dir/src/A.cpp.o"
	cd /home/zhangxk/projects/deepAI/c++/cmake/0b/build/libs/libA && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myA.dir/src/A.cpp.o -c /home/zhangxk/projects/deepAI/c++/cmake/0b/libs/libA/src/A.cpp

libs/libA/CMakeFiles/myA.dir/src/A.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myA.dir/src/A.cpp.i"
	cd /home/zhangxk/projects/deepAI/c++/cmake/0b/build/libs/libA && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhangxk/projects/deepAI/c++/cmake/0b/libs/libA/src/A.cpp > CMakeFiles/myA.dir/src/A.cpp.i

libs/libA/CMakeFiles/myA.dir/src/A.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myA.dir/src/A.cpp.s"
	cd /home/zhangxk/projects/deepAI/c++/cmake/0b/build/libs/libA && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhangxk/projects/deepAI/c++/cmake/0b/libs/libA/src/A.cpp -o CMakeFiles/myA.dir/src/A.cpp.s

# Object files for target myA
myA_OBJECTS = \
"CMakeFiles/myA.dir/src/A.cpp.o"

# External object files for target myA
myA_EXTERNAL_OBJECTS =

libs/libA/libmyA.a: libs/libA/CMakeFiles/myA.dir/src/A.cpp.o
libs/libA/libmyA.a: libs/libA/CMakeFiles/myA.dir/build.make
libs/libA/libmyA.a: libs/libA/CMakeFiles/myA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhangxk/projects/deepAI/c++/cmake/0b/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libmyA.a"
	cd /home/zhangxk/projects/deepAI/c++/cmake/0b/build/libs/libA && $(CMAKE_COMMAND) -P CMakeFiles/myA.dir/cmake_clean_target.cmake
	cd /home/zhangxk/projects/deepAI/c++/cmake/0b/build/libs/libA && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/myA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libs/libA/CMakeFiles/myA.dir/build: libs/libA/libmyA.a

.PHONY : libs/libA/CMakeFiles/myA.dir/build

libs/libA/CMakeFiles/myA.dir/clean:
	cd /home/zhangxk/projects/deepAI/c++/cmake/0b/build/libs/libA && $(CMAKE_COMMAND) -P CMakeFiles/myA.dir/cmake_clean.cmake
.PHONY : libs/libA/CMakeFiles/myA.dir/clean

libs/libA/CMakeFiles/myA.dir/depend:
	cd /home/zhangxk/projects/deepAI/c++/cmake/0b/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhangxk/projects/deepAI/c++/cmake/0b /home/zhangxk/projects/deepAI/c++/cmake/0b/libs/libA /home/zhangxk/projects/deepAI/c++/cmake/0b/build /home/zhangxk/projects/deepAI/c++/cmake/0b/build/libs/libA /home/zhangxk/projects/deepAI/c++/cmake/0b/build/libs/libA/CMakeFiles/myA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libs/libA/CMakeFiles/myA.dir/depend

