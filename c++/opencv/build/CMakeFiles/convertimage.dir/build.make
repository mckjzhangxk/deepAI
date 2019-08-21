# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhangxk/projects/deepAI/c++/opencv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhangxk/projects/deepAI/c++/opencv/build

# Include any dependencies generated for this target.
include CMakeFiles/convertimage.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/convertimage.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/convertimage.dir/flags.make

CMakeFiles/convertimage.dir/convert.cc.o: CMakeFiles/convertimage.dir/flags.make
CMakeFiles/convertimage.dir/convert.cc.o: ../convert.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhangxk/projects/deepAI/c++/opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/convertimage.dir/convert.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/convertimage.dir/convert.cc.o -c /home/zhangxk/projects/deepAI/c++/opencv/convert.cc

CMakeFiles/convertimage.dir/convert.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/convertimage.dir/convert.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhangxk/projects/deepAI/c++/opencv/convert.cc > CMakeFiles/convertimage.dir/convert.cc.i

CMakeFiles/convertimage.dir/convert.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/convertimage.dir/convert.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhangxk/projects/deepAI/c++/opencv/convert.cc -o CMakeFiles/convertimage.dir/convert.cc.s

CMakeFiles/convertimage.dir/convert.cc.o.requires:

.PHONY : CMakeFiles/convertimage.dir/convert.cc.o.requires

CMakeFiles/convertimage.dir/convert.cc.o.provides: CMakeFiles/convertimage.dir/convert.cc.o.requires
	$(MAKE) -f CMakeFiles/convertimage.dir/build.make CMakeFiles/convertimage.dir/convert.cc.o.provides.build
.PHONY : CMakeFiles/convertimage.dir/convert.cc.o.provides

CMakeFiles/convertimage.dir/convert.cc.o.provides.build: CMakeFiles/convertimage.dir/convert.cc.o


# Object files for target convertimage
convertimage_OBJECTS = \
"CMakeFiles/convertimage.dir/convert.cc.o"

# External object files for target convertimage
convertimage_EXTERNAL_OBJECTS =

convertimage: CMakeFiles/convertimage.dir/convert.cc.o
convertimage: CMakeFiles/convertimage.dir/build.make
convertimage: /usr/local/lib/libopencv_shape.so.3.4.6
convertimage: /usr/local/lib/libopencv_objdetect.so.3.4.6
convertimage: /usr/local/lib/libopencv_stitching.so.3.4.6
convertimage: /usr/local/lib/libopencv_videostab.so.3.4.6
convertimage: /usr/local/lib/libopencv_dnn.so.3.4.6
convertimage: /usr/local/lib/libopencv_superres.so.3.4.6
convertimage: /usr/local/lib/libopencv_photo.so.3.4.6
convertimage: /usr/local/lib/libopencv_ml.so.3.4.6
convertimage: /usr/local/lib/libopencv_calib3d.so.3.4.6
convertimage: /usr/local/lib/libopencv_features2d.so.3.4.6
convertimage: /usr/local/lib/libopencv_flann.so.3.4.6
convertimage: /usr/local/lib/libopencv_highgui.so.3.4.6
convertimage: /usr/local/lib/libopencv_videoio.so.3.4.6
convertimage: /usr/local/lib/libopencv_video.so.3.4.6
convertimage: /usr/local/lib/libopencv_imgcodecs.so.3.4.6
convertimage: /usr/local/lib/libopencv_imgproc.so.3.4.6
convertimage: /usr/local/lib/libopencv_core.so.3.4.6
convertimage: CMakeFiles/convertimage.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhangxk/projects/deepAI/c++/opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable convertimage"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convertimage.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/convertimage.dir/build: convertimage

.PHONY : CMakeFiles/convertimage.dir/build

CMakeFiles/convertimage.dir/requires: CMakeFiles/convertimage.dir/convert.cc.o.requires

.PHONY : CMakeFiles/convertimage.dir/requires

CMakeFiles/convertimage.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/convertimage.dir/cmake_clean.cmake
.PHONY : CMakeFiles/convertimage.dir/clean

CMakeFiles/convertimage.dir/depend:
	cd /home/zhangxk/projects/deepAI/c++/opencv/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhangxk/projects/deepAI/c++/opencv /home/zhangxk/projects/deepAI/c++/opencv /home/zhangxk/projects/deepAI/c++/opencv/build /home/zhangxk/projects/deepAI/c++/opencv/build /home/zhangxk/projects/deepAI/c++/opencv/build/CMakeFiles/convertimage.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/convertimage.dir/depend

