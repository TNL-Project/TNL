if( WIN32 )
   # we don't support CUDA on Windows, so we can lower the requirement
   cmake_minimum_required( VERSION 3.23 )
else()
   # cmake 3.24 is required for CUDA with "native" architecture selection
   cmake_minimum_required( VERSION 3.24 )
endif()

project( TNL )

set( tnlVersion "0.1" )

# check if TNL is being used directly or via add_subdirectory
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
   set( TNL_TOP_LEVEL_PROJECT ON )
   set( TNL_SYSTEM "" )
else()
   set( TNL_TOP_LEVEL_PROJECT OFF )
   set( TNL_SYSTEM SYSTEM )
endif()

# declare all custom build options
option( TNL_USE_CUDA "Build with CUDA support" ON )
option( TNL_USE_OPENMP "Build with OpenMP support" ON )
option( TNL_USE_MPI "Build with MPI support" ON )
option( TNL_USE_GMP "Build with GMP support" OFF )
option( TNL_USE_SYSTEM_GTEST "Use GTest installed in the local system and do not download the latest version" OFF )
option( TNL_USE_CI_FLAGS "Add additional compiler flags like -Werror that are enforced in CI builds" OFF )
option( TNL_USE_MARCH_NATIVE_FLAG "Add -march=native and -mtune=native to the list of compiler flags for the Release configuration" OFF )
option( TNL_BUILD_BENCHMARKS "Compile the 'src/Benchmarks' directory" OFF )
option( TNL_BUILD_EXAMPLES "Compile the 'src/Examples' directory" OFF )
option( TNL_BUILD_TOOLS "Compile the 'src/Tools' directory" OFF )
option( TNL_BUILD_TESTS "Build tests" OFF )
option( TNL_BUILD_MATRIX_TESTS "Build tests for matrix formats" OFF )
option( TNL_BUILD_COVERAGE "Enable code coverage reports from unit tests" OFF )
option( TNL_BUILD_DOC "Build examples included in the documentation and generate the documentation with Doxygen" OFF )
option( TNL_OFFLINE_BUILD "Offline build (i.e. without downloading libraries such as GTest)" OFF )

# make cache variables for install destinations
include(GNUInstallDirs)

# install paths relative to the cmake's prefix
set( TNL_TARGET_DATA_DIRECTORY "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}" )
set( TNL_TARGET_CMAKE_DIRECTORY "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" )

# configure build paths
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )

# set cmake's include path so that we can include modules from
# the cmake directory in the TNL repository
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

# Note that in cmake 3.10 the FindOpenMP and FindMPI modules are broken - they do not work when
# CMAKE_EXECUTABLE_SUFFIX is not empty, see https://www.mail-archive.com/cmake@cmake.org/msg56886.html
# and https://gitlab.kitware.com/cmake/cmake/-/issues/23874
# Hence, we find OpenMP and MPI before setting CMAKE_EXECUTABLE_SUFFIX.
find_package( OpenMP COMPONENTS CXX )
find_package( MPI COMPONENTS CXX )
# workaround for https://github.com/ginkgo-project/ginkgo/issues/1096
find_package( Ginkgo QUIET )
# find the std::filesystem library
find_package( Filesystem REQUIRED COMPONENTS Final )

# Settings for debug build
if( CMAKE_BUILD_TYPE STREQUAL "Debug")
    set( CMAKE_EXECUTABLE_SUFFIX "-dbg${CMAKE_EXECUTABLE_SUFFIX}" )  # suffix for executables
    set( CMAKE_DEBUG_POSTFIX "-dbg" )  # suffix for libraries
endif()

# set C++ standard
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# set default build options
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=vla" )
   set( CMAKE_CXX_FLAGS_DEBUG "-g" )
   set( CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" )
endif()
set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}" )

if( TNL_USE_CI_FLAGS )
   # enforce (more or less) warning-free builds
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=uninitialized" )
endif()

# warn about redundant semicolons
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi" )
   if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi-stmt" )
   endif()
endif()

# optimize Release builds for the native CPU arch, unless explicitly disabled
if( TNL_USE_MARCH_NATIVE_FLAG )
   if( CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.0 )
      # -march=native does not work on macOS with clang less than 15.0
      # https://stackoverflow.com/questions/65966969/why-does-march-native-not-work-on-apple-m1
      set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=apple-m1" )
   else()
      set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native" )
   endif()
endif()

# hack needed to avoid "file too big" errors on a 32-bit cross-compiler
# https://stackoverflow.com/questions/71875002/file-too-big-with-mingw-w64-and-cmake
if( MINGW )
   if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
   endif()
   if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       OR ( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC" )
      )
      set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2" )
   endif()
endif()
if( MSVC )
   # "/bigobj" avoids problems with large object files due to heavily templated code
   # "/permissive-" enables two-phase name lookup according to the standard, see https://stackoverflow.com/q/74238513
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /permissive-" )
endif()

if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" )
   # disable some unimportant warnings
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas" )
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
   # Intel's -Wall is very minimalistic, so add -w3 and disable some specific warnings
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w3 -diag-disable:remark" )
endif()

# disable GCC's infamous "maybe-uninitialized" warning (it produces mostly false positives)
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized" )
endif()

# disable false Clang warning: https://stackoverflow.com/q/57645872
if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-self-assign-overloaded" )
endif()

# enable address sanitizer (does not work with MPI due to many false positives, does not work with nvcc at all)
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
   if( NOT TNL_USE_MPI AND NOT TNL_USE_CUDA )
      set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer" )
      set( CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG "${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined" )
      set( CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined" )
      set( CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined" )
   endif()
endif()

# enable link time optimizations (but not in continuous integration)
if( NOT DEFINED ENV{GITLAB_CI} )
   if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
      # FIXME: IPO with GCC 9.1.0 and Debug build = internal compiler error
      # FIXME: IPO with GCC 9.1.0 and nvcc 10.1 and Release build = fatal error: bytecode stream in file `blabla` generated with LTO version 7.1 instead of the expected 8.0
      #set( CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE )
   elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
      # FIXME: IPO with clang from MSYS2 leads to 'file format not recognized' linker errors
      # FIXME: clang does not support IPO for CUDA
      #set( CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE )
   endif()
endif()

# force colorized output in continuous integration
if( DEFINED ENV{GITLAB_CI} OR ${CMAKE_GENERATOR} STREQUAL "Ninja" )
   message(STATUS "Continuous integration or Ninja detected -- forcing compilers to produce colorized output.")
   if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
      set( TNL_COLOR_DIAGNOSTICS_FLAG "-fcolor-diagnostics" )
   elseif( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
      set( TNL_COLOR_DIAGNOSTICS_FLAG "-fdiagnostics-color" )
   endif()
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TNL_COLOR_DIAGNOSTICS_FLAG}" )
endif()

# make cache variable so it can be used in downstream projects
set( TNL_INCLUDE_DIRS
     "${CMAKE_CURRENT_LIST_DIR}/src"
     CACHE INTERNAL "Directories where TNL headers are located" )

# create the exported targets
add_library( TNL INTERFACE )
add_library( TNL_CXX INTERFACE )
add_library( TNL_CUDA INTERFACE )
# aliases to match exported targets
add_library( TNL::TNL ALIAS TNL )
add_library( TNL::TNL_CXX ALIAS TNL_CXX )
add_library( TNL::TNL_CUDA ALIAS TNL_CUDA )
# add dependencies between the exported targets
target_link_libraries( TNL_CXX INTERFACE TNL )
target_link_libraries( TNL_CUDA INTERFACE TNL )

# add the include directory to the interface
target_include_directories( TNL ${TNL_SYSTEM}
                            INTERFACE $<BUILD_INTERFACE:${TNL_INCLUDE_DIRS}>
                                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )
# add compiler features to the interface
target_compile_features( TNL INTERFACE cxx_std_17 )
# add some compiler flags to the interface
target_compile_options( TNL INTERFACE
                        ${TNL_COLOR_DIAGNOSTICS_FLAG} )

# add the std::filesystem library to the interface
target_link_libraries( TNL INTERFACE std::filesystem )

# gtest has to be built before we add the MPI flags
if( TNL_BUILD_TESTS OR TNL_BUILD_MATRIX_TESTS )
   enable_testing()

   # repeat failed tests to tolerate sporadic failures
   set( CMAKE_CTEST_ARGUMENTS ${CMAKE_CTEST_ARGUMENTS} --repeat until-pass:2 )

   # let CTest write test results in the JUnit XML format
   set( CMAKE_CTEST_ARGUMENTS ${CMAKE_CTEST_ARGUMENTS} --output-junit "${CMAKE_BINARY_DIR}/tests-report.xml" )

   if( TNL_USE_SYSTEM_GTEST OR TNL_OFFLINE_BUILD )
      # find gtest installed in the local system
      find_package(GTest REQUIRED)
      if( GTEST_FOUND )
         include_directories( ${GTEST_INCLUDE_DIRS} )
      endif( GTEST_FOUND )
   else()
      # build gtest libs
      include( BuildGtest )
   endif()
   set( CXX_TESTS_FLAGS -DHAVE_GTEST )
   set( CUDA_TESTS_FLAGS -DHAVE_GTEST )
   set( TESTS_LIBRARIES ${GTEST_LIBRARIES} )
   set( TESTS_LINKER_FLAGS "" )

   if( TNL_BUILD_COVERAGE AND CMAKE_BUILD_TYPE STREQUAL "Debug" )
      # set compiler flags needed for code coverage
      set( CXX_TESTS_FLAGS ${CXX_TESTS_FLAGS} --coverage )
      set( CUDA_TESTS_FLAGS ${CUDA_TESTS_FLAGS} -Xcompiler --coverage )
      set( TESTS_LINKER_FLAGS ${TESTS_LINKER_FLAGS} --coverage )
   endif()
endif()

####
# Check for threads library
#
set( CMAKE_THREAD_PREFER_PTHREAD TRUE )
set( THREADS_PREFER_PTHREAD_FLAG TRUE )
find_package( Threads REQUIRED )
target_link_libraries( TNL_CXX INTERFACE Threads::Threads )

####
# Check for OpenMP
#
if( OPENMP_FOUND AND TNL_USE_OPENMP )
   # NOTE: OpenMP + CUDA = segfault in matrix tests (the host code built by nvcc is also not optimal)
   target_compile_definitions( TNL_CXX INTERFACE HAVE_OPENMP )
   target_link_libraries( TNL_CXX INTERFACE OpenMP::OpenMP_CXX )
   set( TNL_BUILD_OPENMP TRUE )
endif()

####
# Check for MPI
#
if( MPI_CXX_FOUND AND TNL_USE_MPI )
   set( TNL_BUILD_MPI TRUE )
   # NOTE: MPI will be added selectively to the targets that need/can utilize it
   # see https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
endif()

#####
# Check for CUDA
#
if( TNL_USE_CUDA )
   include( CheckLanguage )
   # check_language(CUDA) or enable_language(CUDA) does not work with CMAKE_EXECUTABLE_SUFFIX
   # see https://gitlab.kitware.com/cmake/cmake/-/issues/23887
   set( executable_suffix_backup "${CMAKE_EXECUTABLE_SUFFIX}" )
   set( CMAKE_EXECUTABLE_SUFFIX "" )
   check_language( CUDA )
   if( CMAKE_CUDA_COMPILER )
      enable_language( CUDA )
      set( TNL_BUILD_CUDA TRUE)

      # set C++ standard
      set( CMAKE_CUDA_STANDARD 17 )
      set( CMAKE_CUDA_STANDARD_REQUIRED ON )
      set( CMAKE_CUDA_EXTENSIONS OFF )

      # enable separable compilation
      set( CMAKE_CUDA_SEPARABLE_COMPILATION ON )

      # set necessary CUDA compiler flags on the interface
      target_compile_options( TNL_CUDA INTERFACE
            $<$<CUDA_COMPILER_ID:NVIDIA>:
                  --expt-relaxed-constexpr ;
                  --extended-lambda ;
                  --default-stream per-thread ;
            >
      )

      # Disable false compiler warnings
      #   reference for the --diag_suppress and --display_error_number flags: https://stackoverflow.com/a/54142937
      #   incomplete list of tokens: http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
      target_compile_options( TNL_CUDA INTERFACE
            $<$<CUDA_COMPILER_ID:NVIDIA>:
                  -Wno-deprecated-gpu-targets ;
                  "SHELL:-Xcudafe --diag_suppress=code_is_unreachable" ;
                  "SHELL:-Xcudafe --diag_suppress=loop_not_reachable" ;
                  "SHELL:-Xcudafe --diag_suppress=implicit_return_from_non_void_function" ;
                  "SHELL:-Xcudafe --diag_suppress=unsigned_compare_with_zero" ;
                  --display-error-number ;
            >
      )
      # This diagnostic is just plain wrong in CUDA 9 and later, see https://github.com/kokkos/kokkos/issues/1470
      target_compile_options( TNL_CUDA INTERFACE
            $<$<CUDA_COMPILER_ID:NVIDIA>:
                  "SHELL:-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored"
            >
      )
      # nvcc 10 causes many invalid VLA errors in the host code
      target_compile_options( TNL_CUDA INTERFACE
            $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<VERSION_LESS:$<CUDA_COMPILER_VERSION>,11>>:
                  "SHELL:-Xcompiler -Wno-vla"
            >
      )

      # set project-specific (i.e. not exported) build options
      set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall" )
      set( CMAKE_CUDA_FLAGS_DEBUG "-g" )
      set( CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG" )
      set( CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELEASE} ${CMAKE_CUDA_FLAGS_DEBUG} --generate-line-info" )

      if( CMAKE_CUDA_COMPILER_ID STREQUAL "Clang" )
         # disable some unimportant warnings
         set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-pragmas" )
         if( TNL_USE_CI_FLAGS )
            # enforce (more or less) warning-free builds
            set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=unknown-cuda-version" )
         endif()
         # workaround for Clang 15
         # https://github.com/llvm/llvm-project/issues/58491
         set( CMAKE_CUDA_FLAGS_DEBUG "-g -Xarch_device -g0" )
      endif()
   endif()
   set( CMAKE_EXECUTABLE_SUFFIX "${executable_suffix_backup}" )
   unset( executable_suffix_backup )
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if( DEFINED ENV{GITLAB_CI} AND TNL_BUILD_CUDA )
   set( TNL_BUILD_CPP_TARGETS FALSE )
else()
   set( TNL_BUILD_CPP_TARGETS TRUE )
endif()


find_package( DCMTK )
if( DCMTK_FOUND )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_DCMTK_H" )
   include_directories( ${DCMTK_INCLUDE_DIRS} )
endif()

find_package( PNG )
if( PNG_FOUND )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_PNG_H" )
   include_directories( ${PNG_INCLUDE_DIRS} )
endif()

find_package( JPEG )
if( JPEG_FOUND )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_JPEG_H" )
   include_directories( ${JPEG_INCLUDE_DIRS} )
endif()

####
# Test for GMP
#
if( TNL_USE_GMP )
   if (GMP_INCLUDES AND GMP_LIBRARIES)
      set(GMP_FIND_QUIETLY TRUE)
   endif (GMP_INCLUDES AND GMP_LIBRARIES)

   find_path(GMP_INCLUDES
      NAMES
      gmp.h
      PATHS
      $ENV{GMPDIR}
      ${INCLUDE_INSTALL_DIR}
   )

   find_library(GMP_LIBRARIES gmp PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

   include(FindPackageHandleStandardArgs)
   find_package_handle_standard_args(GMP DEFAULT_MSG
                                      GMP_INCLUDES GMP_LIBRARIES)
   if( GMP_INCLUDES STREQUAL "GMP_INCLUDES-NOTFOUND" OR GMP_LIBRARIES STREQUAL "GMP_LIBRARIES-NOTFOUND" )
      message( "GMP was not found. Some tests for higher precision arithmetics will not be passed." )
   else()
      set( HAVE_GMP )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${GMP_INCLUDES} -DHAVE_GMP" )
      set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${GMP_LIBRARIES}" )
      mark_as_advanced(GMP_INCLUDES GMP_LIBRARIES)
   endif()
endif()

if( TNL_TOP_LEVEL_PROJECT )
   # export the interface targets
   install( TARGETS TNL TNL_CXX TNL_CUDA EXPORT TNLTargets )
   # install a CMake file for the interface target
   install( EXPORT TNLTargets
            NAMESPACE TNL::
            DESTINATION ${TNL_TARGET_CMAKE_DIRECTORY} )
   # install the TNLConfig.cmake file
   include( CMakePackageConfigHelpers )
   configure_package_config_file( "cmake/${PROJECT_NAME}Config.cmake.in"
                                  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                                  INSTALL_DESTINATION ${TNL_TARGET_CMAKE_DIRECTORY} )
   install( FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
            DESTINATION ${TNL_TARGET_CMAKE_DIRECTORY} )
   # install necessary modules
   install( FILES cmake/FindFilesystem.cmake
            DESTINATION ${TNL_TARGET_CMAKE_DIRECTORY}/Modules )
endif()

# Add all subdirectories
add_subdirectory( src )
add_subdirectory( Documentation )

# export compile_commands.json so it can be used by Clang tools
# https://clang.llvm.org/docs/JSONCompilationDatabase.html
set( CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "" )

if( TNL_TOP_LEVEL_PROJECT )
   # Print custom build options
   message( "-- Build options:" )
   message( "   TNL_USE_CUDA = ${TNL_USE_CUDA}" )
   message( "   CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}" )
   message( "   TNL_USE_OPENMP = ${TNL_USE_OPENMP}" )
   message( "   TNL_USE_MPI = ${TNL_USE_MPI}" )
   message( "   TNL_USE_GMP = ${TNL_USE_GMP}" )
   message( "   TNL_USE_SYSTEM_GTEST= ${TNL_USE_SYSTEM_GTEST}" )
   message( "   TNL_BUILD_BENCHMARKS = ${TNL_BUILD_BENCHMARKS}" )
   message( "   TNL_BUILD_EXAMPLES = ${TNL_BUILD_EXAMPLES}" )
   message( "   TNL_BUILD_TOOLS = ${TNL_BUILD_TOOLS}" )
   message( "   TNL_BUILD_TESTS = ${TNL_BUILD_TESTS}" )
   message( "   TNL_BUILD_MATRIX_TESTS = ${TNL_BUILD_MATRIX_TESTS}" )
   message( "   TNL_BUILD_COVERAGE = ${TNL_BUILD_COVERAGE}" )
   message( "   TNL_BUILD_DOC = ${TNL_BUILD_DOC}" )
   message( "   TNL_OFFLINE_BUILD = ${TNL_OFFLINE_BUILD}" )
   # Print compiler options
   message( "-- Compiler options:" )
   message( "   CMAKE_C_FLAGS = ${CMAKE_C_FLAGS}" )
   message( "   CMAKE_C_FLAGS_DEBUG = ${CMAKE_C_FLAGS_DEBUG}" )
   message( "   CMAKE_C_FLAGS_RELEASE = ${CMAKE_C_FLAGS_RELEASE}" )
   message( "   CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}" )
   message( "   CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}" )
   message( "   CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}" )
   message( "   CMAKE_CUDA_FLAGS = ${CMAKE_CUDA_FLAGS}" )
   message( "   CMAKE_CUDA_FLAGS_DEBUG = ${CMAKE_CUDA_FLAGS_DEBUG}" )
   message( "   CMAKE_CUDA_FLAGS_RELEASE = ${CMAKE_CUDA_FLAGS_RELEASE}" )
   message( "   CMAKE_SHARED_LIBRARY_LINK_C_FLAGS = ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS}" )
   message( "   CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG = ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG}" )
   message( "   CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_RELEASE = ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_RELEASE}" )
   message( "   CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}" )
   message( "   CMAKE_EXE_LINKER_FLAGS_DEBUG = ${CMAKE_EXE_LINKER_FLAGS_DEBUG}" )
   message( "   CMAKE_EXE_LINKER_FLAGS_RELEASE = ${CMAKE_EXE_LINKER_FLAGS_RELEASE}" )
   message( "   CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}" )
   message( "   CMAKE_SHARED_LINKER_FLAGS_DEBUG = ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" )
   message( "   CMAKE_SHARED_LINKER_FLAGS_RELEASE = ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" )
   message( "   GMP_LIBRARIES = ${GMP_LIBRARIES}" )

   if( MPI_CXX_FOUND AND TNL_USE_MPI )
      message( "   MPI_CXX_COMPILE_OPTIONS = ${MPI_CXX_COMPILE_OPTIONS}" )
      message( "   MPI_CXX_COMPILE_DEFINITIONS = ${MPI_CXX_COMPILE_DEFINITIONS}" )
      message( "   MPI_CXX_INCLUDE_DIRS = ${MPI_CXX_INCLUDE_DIRS}" )
      message( "   MPI_CXX_LINK_FLAGS = ${MPI_CXX_LINK_FLAGS}" )
      message( "   MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}" )
   endif()
endif()
