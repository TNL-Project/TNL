# cmake 3.24 is required for CUDA with "native" architecture selection
# cmake 3.25 is required for the "SYSTEM" keyword in FetchContent_Declare
cmake_minimum_required( VERSION 3.25 )

project( TNL )

set( tnlVersion "0.1" )

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

# Optionally bundled dependencies (system packages are preferred)
# Reference: https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html#fetchcontent-and-find-package-integration
include( FetchTinyXML2 )

# Settings for debug build
if( CMAKE_BUILD_TYPE STREQUAL "Debug")
    set( CMAKE_EXECUTABLE_SUFFIX "-dbg${CMAKE_EXECUTABLE_SUFFIX}" )  # suffix for executables
    set( CMAKE_DEBUG_POSTFIX "-dbg" )  # suffix for libraries
endif()

# set C++ standard
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

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
if( PROJECT_IS_TOP_LEVEL )
   set( TNL_SYSTEM "" )
else()
   set( TNL_SYSTEM SYSTEM )
endif()
target_include_directories( TNL ${TNL_SYSTEM}
                            INTERFACE $<BUILD_INTERFACE:${TNL_INCLUDE_DIRS}>
                                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )
# add compiler features to the interface
target_compile_features( TNL INTERFACE cxx_std_17 )

# add the std::filesystem library to the interface
target_link_libraries( TNL INTERFACE std::filesystem )

# set C++ compiler flags
include( cmake/cxx_flags.cmake )

# gtest has to be built before we add the MPI flags
if( TNL_BUILD_TESTS OR TNL_BUILD_MATRIX_TESTS )
   enable_testing()

   # repeat failed tests to tolerate sporadic failures
   set( CMAKE_CTEST_ARGUMENTS ${CMAKE_CTEST_ARGUMENTS} --repeat until-pass:2 )

   # let CTest write test results in the JUnit XML format
   set( CMAKE_CTEST_ARGUMENTS ${CMAKE_CTEST_ARGUMENTS} --output-junit "${CMAKE_BINARY_DIR}/tests-report.xml" )

   if( TNL_USE_SYSTEM_GTEST OR TNL_OFFLINE_BUILD )
      # find GoogleTest installed in the local system
      find_package( GTest REQUIRED )
   else()
      # fetch and build GoogleTest from source
      include( FetchGoogleTest )
   endif()
   set( CXX_TESTS_FLAGS )
   set( CUDA_TESTS_FLAGS )
   set( TESTS_LIBRARIES GTest::gtest_main )
   set( TESTS_LINKER_FLAGS "" )

   if( TNL_BUILD_COVERAGE AND CMAKE_BUILD_TYPE STREQUAL "Debug" )
      # set compiler flags needed for code coverage
      set( CXX_TESTS_FLAGS ${CXX_TESTS_FLAGS} --coverage )
      set( CUDA_TESTS_FLAGS ${CUDA_TESTS_FLAGS} -Xcompiler --coverage )
      set( TESTS_LINKER_FLAGS ${TESTS_LINKER_FLAGS} --coverage )
   endif()

   # function to simplify adding MPI tests
   # usage: add_test_mpi(NAME <name> NPROC <nproc> COMMAND <command> [<arg>...])
   function( add_test_mpi )
      # parse function arguments
      set( options )
      set( oneValueArgs NAME NPROC )
      set( multiValueArgs COMMAND )
      cmake_parse_arguments( ADD_TEST_MPI "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
      # set flags for MPIEXEC_EXECUTABLE
      set( mpiexec_flags "${MPIEXEC_NUMPROC_FLAG}" "${ADD_TEST_MPI_NPROC}" -H "localhost:${ADD_TEST_MPI_NPROC}" )
      # set OMP_NUM_THREADS=1 to disable OpenMP for MPI tests
      # (NPROC may be even greater than the number of physical cores, so it would just slow down)
      add_test( NAME ${ADD_TEST_MPI_NAME} COMMAND "env" "OMP_NUM_THREADS=1" "${MPIEXEC_EXECUTABLE}" ${mpiexec_flags} ${ADD_TEST_MPI_COMMAND} )
   endfunction()
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
      set( TNL_BUILD_CUDA TRUE )

      # set C++ standard
      set( CMAKE_CUDA_STANDARD 17 )
      set( CMAKE_CUDA_STANDARD_REQUIRED ON )
      set( CMAKE_CUDA_EXTENSIONS OFF )

      # enable separable compilation
      set( CMAKE_CUDA_SEPARABLE_COMPILATION ON )

      # set CUDA compiler flags
      include( cmake/cuda_flags.cmake )
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

if( PROJECT_IS_TOP_LEVEL )
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

if( PROJECT_IS_TOP_LEVEL )
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
