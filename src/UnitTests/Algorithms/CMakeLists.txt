ADD_SUBDIRECTORY( Segments )
ADD_SUBDIRECTORY( Sorting )

set( COMMON_TESTS
         containsTest
         copyTest
         equalTest
         fillTest
         findTest
         MultireductionTest
         parallelForTest
         staticForTest
         unrolledForTest
)

set( CPP_TESTS
         reduceTest
         scanTest
         SegmentedScanTest
)
set( CUDA_TESTS
         reduceTestCuda
         scanTestCuda
)
if( TNL_BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} ${COMMON_TESTS} )
else()
   set( CPP_TESTS  ${CPP_TESTS} ${COMMON_TESTS} )
endif()

foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )
   add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
endforeach()

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_TESTS} )
      add_executable( ${target} ${target}.cu )
      target_compile_options( ${target} PUBLIC ${CUDA_TESTS_FLAGS} )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
      target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
endif()


if( TNL_BUILD_MPI )
   add_executable( distributedScanTest distributedScanTest.cpp )
   target_compile_options( distributedScanTest PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( distributedScanTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( distributedScanTest PUBLIC ${TESTS_LINKER_FLAGS} )
   # enable MPI support in TNL
   target_compile_definitions( distributedScanTest PUBLIC "-DHAVE_MPI" )
   # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
   target_link_libraries( distributedScanTest PUBLIC MPI::MPI_CXX )

   if( TNL_BUILD_CUDA )
      add_executable( distributedScanTestCuda distributedScanTestCuda.cu )
      target_compile_options( distributedScanTestCuda PUBLIC ${CUDA_TESTS_FLAGS} )
      target_link_libraries( distributedScanTestCuda PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
      target_link_options( distributedScanTestCuda PUBLIC ${TESTS_LINKER_FLAGS} )
      # enable MPI support in TNL
      target_compile_definitions( distributedScanTestCuda PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( distributedScanTestCuda PUBLIC MPI::MPI_CXX )
   endif()

   add_test_mpi( NAME distributedScanTest NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTest${CMAKE_EXECUTABLE_SUFFIX}" )
   add_test_mpi( NAME distributedScanTest_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTest${CMAKE_EXECUTABLE_SUFFIX}" )

   if( TNL_BUILD_CUDA )
      add_test_mpi( NAME distributedScanTestCuda NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      add_test_mpi( NAME distributedScanTestCuda_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
   endif()
endif()
