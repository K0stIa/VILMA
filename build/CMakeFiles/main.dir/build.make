# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.2.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.2.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Shared/research/code/vilma

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Shared/research/code/vilma/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/Shared/research/code/vilma/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/main.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main.cpp.o -c /Users/Shared/research/code/vilma/main.cpp

CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/Shared/research/code/vilma/main.cpp > CMakeFiles/main.dir/main.cpp.i

CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/Shared/research/code/vilma/main.cpp -o CMakeFiles/main.dir/main.cpp.s

CMakeFiles/main.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/main.dir/main.cpp.o.requires

CMakeFiles/main.dir/main.cpp.o.provides: CMakeFiles/main.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/main.cpp.o.provides

CMakeFiles/main.dir/main.cpp.o.provides.build: CMakeFiles/main.dir/main.cpp.o

CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o: ../bmrm/bmrm_solver.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/Shared/research/code/vilma/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o -c /Users/Shared/research/code/vilma/bmrm/bmrm_solver.cpp

CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/Shared/research/code/vilma/bmrm/bmrm_solver.cpp > CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.i

CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/Shared/research/code/vilma/bmrm/bmrm_solver.cpp -o CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.s

CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o.requires:
.PHONY : CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o.requires

CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o.provides: CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o.provides

CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o.provides.build: CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o

CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o: ../bmrm/libqp_splx.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/Shared/research/code/vilma/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o -c /Users/Shared/research/code/vilma/bmrm/libqp_splx.cpp

CMakeFiles/main.dir/bmrm/libqp_splx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/bmrm/libqp_splx.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/Shared/research/code/vilma/bmrm/libqp_splx.cpp > CMakeFiles/main.dir/bmrm/libqp_splx.cpp.i

CMakeFiles/main.dir/bmrm/libqp_splx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/bmrm/libqp_splx.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/Shared/research/code/vilma/bmrm/libqp_splx.cpp -o CMakeFiles/main.dir/bmrm/libqp_splx.cpp.s

CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o.requires:
.PHONY : CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o.requires

CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o.provides: CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o.provides

CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o.provides.build: CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o

CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o: ../oracle/single_gender_no_beta_bmrm_oracle.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/Shared/research/code/vilma/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o -c /Users/Shared/research/code/vilma/oracle/single_gender_no_beta_bmrm_oracle.cpp

CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/Shared/research/code/vilma/oracle/single_gender_no_beta_bmrm_oracle.cpp > CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.i

CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/Shared/research/code/vilma/oracle/single_gender_no_beta_bmrm_oracle.cpp -o CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.s

CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o.requires:
.PHONY : CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o.requires

CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o.provides: CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o.provides

CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o.provides.build: CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o

CMakeFiles/main.dir/data.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/data.cpp.o: ../data.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/Shared/research/code/vilma/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/main.dir/data.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/main.dir/data.cpp.o -c /Users/Shared/research/code/vilma/data.cpp

CMakeFiles/main.dir/data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/data.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/Shared/research/code/vilma/data.cpp > CMakeFiles/main.dir/data.cpp.i

CMakeFiles/main.dir/data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/data.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/Shared/research/code/vilma/data.cpp -o CMakeFiles/main.dir/data.cpp.s

CMakeFiles/main.dir/data.cpp.o.requires:
.PHONY : CMakeFiles/main.dir/data.cpp.o.requires

CMakeFiles/main.dir/data.cpp.o.provides: CMakeFiles/main.dir/data.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/data.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/data.cpp.o.provides

CMakeFiles/main.dir/data.cpp.o.provides.build: CMakeFiles/main.dir/data.cpp.o

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o" \
"CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o" \
"CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o" \
"CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o" \
"CMakeFiles/main.dir/data.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/main.cpp.o
main: CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o
main: CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o
main: CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o
main: CMakeFiles/main.dir/data.cpp.o
main: CMakeFiles/main.dir/build.make
main: /Users/Shared/research/code/python/jmlr/oboe/lib/liboboe.a
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/requires: CMakeFiles/main.dir/main.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/bmrm/bmrm_solver.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/bmrm/libqp_splx.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/oracle/single_gender_no_beta_bmrm_oracle.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/data.cpp.o.requires
.PHONY : CMakeFiles/main.dir/requires

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /Users/Shared/research/code/vilma/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Shared/research/code/vilma /Users/Shared/research/code/vilma /Users/Shared/research/code/vilma/build /Users/Shared/research/code/vilma/build /Users/Shared/research/code/vilma/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

