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
CMAKE_SOURCE_DIR = /home/zhangxk/projects/deepAI/c++/computerGraph/three

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhangxk/projects/deepAI/c++/computerGraph/three/build

# Include any dependencies generated for this target.
include CMakeFiles/a3.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/a3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/a3.dir/flags.make

CMakeFiles/a3.dir/ClothSystem.cpp.o: CMakeFiles/a3.dir/flags.make
CMakeFiles/a3.dir/ClothSystem.cpp.o: ../ClothSystem.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhangxk/projects/deepAI/c++/computerGraph/three/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/a3.dir/ClothSystem.cpp.o"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/a3.dir/ClothSystem.cpp.o -c /home/zhangxk/projects/deepAI/c++/computerGraph/three/ClothSystem.cpp

CMakeFiles/a3.dir/ClothSystem.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/a3.dir/ClothSystem.cpp.i"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhangxk/projects/deepAI/c++/computerGraph/three/ClothSystem.cpp > CMakeFiles/a3.dir/ClothSystem.cpp.i

CMakeFiles/a3.dir/ClothSystem.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/a3.dir/ClothSystem.cpp.s"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhangxk/projects/deepAI/c++/computerGraph/three/ClothSystem.cpp -o CMakeFiles/a3.dir/ClothSystem.cpp.s

CMakeFiles/a3.dir/TimeStepper.cpp.o: CMakeFiles/a3.dir/flags.make
CMakeFiles/a3.dir/TimeStepper.cpp.o: ../TimeStepper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhangxk/projects/deepAI/c++/computerGraph/three/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/a3.dir/TimeStepper.cpp.o"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/a3.dir/TimeStepper.cpp.o -c /home/zhangxk/projects/deepAI/c++/computerGraph/three/TimeStepper.cpp

CMakeFiles/a3.dir/TimeStepper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/a3.dir/TimeStepper.cpp.i"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhangxk/projects/deepAI/c++/computerGraph/three/TimeStepper.cpp > CMakeFiles/a3.dir/TimeStepper.cpp.i

CMakeFiles/a3.dir/TimeStepper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/a3.dir/TimeStepper.cpp.s"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhangxk/projects/deepAI/c++/computerGraph/three/TimeStepper.cpp -o CMakeFiles/a3.dir/TimeStepper.cpp.s

CMakeFiles/a3.dir/camera.cpp.o: CMakeFiles/a3.dir/flags.make
CMakeFiles/a3.dir/camera.cpp.o: ../camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhangxk/projects/deepAI/c++/computerGraph/three/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/a3.dir/camera.cpp.o"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/a3.dir/camera.cpp.o -c /home/zhangxk/projects/deepAI/c++/computerGraph/three/camera.cpp

CMakeFiles/a3.dir/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/a3.dir/camera.cpp.i"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhangxk/projects/deepAI/c++/computerGraph/three/camera.cpp > CMakeFiles/a3.dir/camera.cpp.i

CMakeFiles/a3.dir/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/a3.dir/camera.cpp.s"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhangxk/projects/deepAI/c++/computerGraph/three/camera.cpp -o CMakeFiles/a3.dir/camera.cpp.s

CMakeFiles/a3.dir/main.cpp.o: CMakeFiles/a3.dir/flags.make
CMakeFiles/a3.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhangxk/projects/deepAI/c++/computerGraph/three/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/a3.dir/main.cpp.o"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/a3.dir/main.cpp.o -c /home/zhangxk/projects/deepAI/c++/computerGraph/three/main.cpp

CMakeFiles/a3.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/a3.dir/main.cpp.i"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhangxk/projects/deepAI/c++/computerGraph/three/main.cpp > CMakeFiles/a3.dir/main.cpp.i

CMakeFiles/a3.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/a3.dir/main.cpp.s"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhangxk/projects/deepAI/c++/computerGraph/three/main.cpp -o CMakeFiles/a3.dir/main.cpp.s

CMakeFiles/a3.dir/particleSystem.cpp.o: CMakeFiles/a3.dir/flags.make
CMakeFiles/a3.dir/particleSystem.cpp.o: ../particleSystem.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhangxk/projects/deepAI/c++/computerGraph/three/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/a3.dir/particleSystem.cpp.o"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/a3.dir/particleSystem.cpp.o -c /home/zhangxk/projects/deepAI/c++/computerGraph/three/particleSystem.cpp

CMakeFiles/a3.dir/particleSystem.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/a3.dir/particleSystem.cpp.i"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhangxk/projects/deepAI/c++/computerGraph/three/particleSystem.cpp > CMakeFiles/a3.dir/particleSystem.cpp.i

CMakeFiles/a3.dir/particleSystem.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/a3.dir/particleSystem.cpp.s"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhangxk/projects/deepAI/c++/computerGraph/three/particleSystem.cpp -o CMakeFiles/a3.dir/particleSystem.cpp.s

CMakeFiles/a3.dir/pendulumSystem.cpp.o: CMakeFiles/a3.dir/flags.make
CMakeFiles/a3.dir/pendulumSystem.cpp.o: ../pendulumSystem.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhangxk/projects/deepAI/c++/computerGraph/three/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/a3.dir/pendulumSystem.cpp.o"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/a3.dir/pendulumSystem.cpp.o -c /home/zhangxk/projects/deepAI/c++/computerGraph/three/pendulumSystem.cpp

CMakeFiles/a3.dir/pendulumSystem.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/a3.dir/pendulumSystem.cpp.i"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhangxk/projects/deepAI/c++/computerGraph/three/pendulumSystem.cpp > CMakeFiles/a3.dir/pendulumSystem.cpp.i

CMakeFiles/a3.dir/pendulumSystem.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/a3.dir/pendulumSystem.cpp.s"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhangxk/projects/deepAI/c++/computerGraph/three/pendulumSystem.cpp -o CMakeFiles/a3.dir/pendulumSystem.cpp.s

CMakeFiles/a3.dir/simpleSystem.cpp.o: CMakeFiles/a3.dir/flags.make
CMakeFiles/a3.dir/simpleSystem.cpp.o: ../simpleSystem.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhangxk/projects/deepAI/c++/computerGraph/three/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/a3.dir/simpleSystem.cpp.o"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/a3.dir/simpleSystem.cpp.o -c /home/zhangxk/projects/deepAI/c++/computerGraph/three/simpleSystem.cpp

CMakeFiles/a3.dir/simpleSystem.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/a3.dir/simpleSystem.cpp.i"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhangxk/projects/deepAI/c++/computerGraph/three/simpleSystem.cpp > CMakeFiles/a3.dir/simpleSystem.cpp.i

CMakeFiles/a3.dir/simpleSystem.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/a3.dir/simpleSystem.cpp.s"
	/usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhangxk/projects/deepAI/c++/computerGraph/three/simpleSystem.cpp -o CMakeFiles/a3.dir/simpleSystem.cpp.s

# Object files for target a3
a3_OBJECTS = \
"CMakeFiles/a3.dir/ClothSystem.cpp.o" \
"CMakeFiles/a3.dir/TimeStepper.cpp.o" \
"CMakeFiles/a3.dir/camera.cpp.o" \
"CMakeFiles/a3.dir/main.cpp.o" \
"CMakeFiles/a3.dir/particleSystem.cpp.o" \
"CMakeFiles/a3.dir/pendulumSystem.cpp.o" \
"CMakeFiles/a3.dir/simpleSystem.cpp.o"

# External object files for target a3
a3_EXTERNAL_OBJECTS =

a3: CMakeFiles/a3.dir/ClothSystem.cpp.o
a3: CMakeFiles/a3.dir/TimeStepper.cpp.o
a3: CMakeFiles/a3.dir/camera.cpp.o
a3: CMakeFiles/a3.dir/main.cpp.o
a3: CMakeFiles/a3.dir/particleSystem.cpp.o
a3: CMakeFiles/a3.dir/pendulumSystem.cpp.o
a3: CMakeFiles/a3.dir/simpleSystem.cpp.o
a3: CMakeFiles/a3.dir/build.make
a3: CMakeFiles/a3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhangxk/projects/deepAI/c++/computerGraph/three/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable a3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/a3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/a3.dir/build: a3

.PHONY : CMakeFiles/a3.dir/build

CMakeFiles/a3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/a3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/a3.dir/clean

CMakeFiles/a3.dir/depend:
	cd /home/zhangxk/projects/deepAI/c++/computerGraph/three/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhangxk/projects/deepAI/c++/computerGraph/three /home/zhangxk/projects/deepAI/c++/computerGraph/three /home/zhangxk/projects/deepAI/c++/computerGraph/three/build /home/zhangxk/projects/deepAI/c++/computerGraph/three/build /home/zhangxk/projects/deepAI/c++/computerGraph/three/build/CMakeFiles/a3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/a3.dir/depend

