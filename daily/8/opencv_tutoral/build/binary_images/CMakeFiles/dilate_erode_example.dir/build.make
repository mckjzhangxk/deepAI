# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build

# Include any dependencies generated for this target.
include binary_images/CMakeFiles/dilate_erode_example.dir/depend.make

# Include the progress variables for this target.
include binary_images/CMakeFiles/dilate_erode_example.dir/progress.make

# Include the compile flags for this target's objects.
include binary_images/CMakeFiles/dilate_erode_example.dir/flags.make

binary_images/CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.o: binary_images/CMakeFiles/dilate_erode_example.dir/flags.make
binary_images/CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.o: ../binary_images/dilate_erode_example.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object binary_images/CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.o"
	cd /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build/binary_images && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.o -c /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/binary_images/dilate_erode_example.cpp

binary_images/CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.i"
	cd /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build/binary_images && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/binary_images/dilate_erode_example.cpp > CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.i

binary_images/CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.s"
	cd /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build/binary_images && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/binary_images/dilate_erode_example.cpp -o CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.s

# Object files for target dilate_erode_example
dilate_erode_example_OBJECTS = \
"CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.o"

# External object files for target dilate_erode_example
dilate_erode_example_EXTERNAL_OBJECTS =

../bin/dilate_erode_example: binary_images/CMakeFiles/dilate_erode_example.dir/dilate_erode_example.cpp.o
../bin/dilate_erode_example: binary_images/CMakeFiles/dilate_erode_example.dir/build.make
../bin/dilate_erode_example: /usr/local/lib/libopencv_stitching.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_ml.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_videostab.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_superres.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_shape.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_objdetect.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_dnn.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_calib3d.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_features2d.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_photo.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_flann.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_highgui.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_videoio.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_imgcodecs.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_video.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_imgproc.so.3.4.2
../bin/dilate_erode_example: /usr/local/lib/libopencv_core.so.3.4.2
../bin/dilate_erode_example: binary_images/CMakeFiles/dilate_erode_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/dilate_erode_example"
	cd /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build/binary_images && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dilate_erode_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
binary_images/CMakeFiles/dilate_erode_example.dir/build: ../bin/dilate_erode_example

.PHONY : binary_images/CMakeFiles/dilate_erode_example.dir/build

binary_images/CMakeFiles/dilate_erode_example.dir/clean:
	cd /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build/binary_images && $(CMAKE_COMMAND) -P CMakeFiles/dilate_erode_example.dir/cmake_clean.cmake
.PHONY : binary_images/CMakeFiles/dilate_erode_example.dir/clean

binary_images/CMakeFiles/dilate_erode_example.dir/depend:
	cd /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/binary_images /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build/binary_images /home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/build/binary_images/CMakeFiles/dilate_erode_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : binary_images/CMakeFiles/dilate_erode_example.dir/depend

