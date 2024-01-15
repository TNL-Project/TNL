# include CTest only when the project is top-level (not when it is added to
# the build tree of another project with add_subdirectory)
if( PROJECT_IS_TOP_LEVEL )
   # configure the project for testing with CTest/CDash
   include( CTest )

   if( TNL_USE_SYSTEM_GTEST OR TNL_OFFLINE_BUILD )
      # find GoogleTest installed in the local system
      find_package( GTest REQUIRED )
   else()
      # fetch and build GoogleTest from source
      include( FetchGoogleTest )
   endif()
   set( CXX_TESTS_FLAGS )
   set( CUDA_TESTS_FLAGS )
   set( HIP_TESTS_FLAGS )
   set( TESTS_LIBRARIES GTest::gtest_main )
   set( TESTS_LINKER_FLAGS "" )

   if( TNL_BUILD_COVERAGE AND CMAKE_BUILD_TYPE STREQUAL "Debug" )
      # set compiler flags needed for code coverage
      set( CXX_TESTS_FLAGS ${CXX_TESTS_FLAGS} --coverage )
      set( CUDA_TESTS_FLAGS ${CUDA_TESTS_FLAGS} -Xcompiler --coverage )
      set( HIP_TESTS_FLAGS ${HIP_TESTS_FLAGS} --coverage )
      set( TESTS_LINKER_FLAGS ${TESTS_LINKER_FLAGS} --coverage )
   endif()
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

   add_test( NAME ${ADD_TEST_MPI_NAME} COMMAND "${MPIEXEC_EXECUTABLE}" ${mpiexec_flags} ${ADD_TEST_MPI_COMMAND} )

   # set OMP_NUM_THREADS=1 to disable OpenMP for MPI tests
   # (NPROC may be even greater than the number of physical cores, so it would just slow down)
   set_property( TEST ${ADD_TEST_MPI_NAME} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1" )

   # add "MPI" label to the test
   set_property( TEST ${ADD_TEST_MPI_NAME} PROPERTY LABELS MPI )

   # set the number of processes used by the test
   set_property( TEST ${ADD_TEST_MPI_NAME} PROPERTY PROCESSORS ${ADD_TEST_MPI_NPROC} )
endfunction()
