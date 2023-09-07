ADD_SUBDIRECTORY( ndarray )

set( CPP_TESTS
         MemoryOperationsTest
         ArrayTest
         ArrayViewTest
         StaticArrayTest
         StaticVectorTest
         StaticVectorOperationsTest
         StaticVectorOfStaticVectorsTest
         VectorTest
         VectorEvaluateAndReduceTest
         VectorBinaryOperationsTest
         VectorUnaryOperationsTest
         VectorVerticalOperationsTest
         VectorOfStaticVectorsTest
)
set( CUDA_TESTS
         MemoryOperationsTestCuda
         ArrayTestCuda
         ArrayViewTestCuda
         VectorTestCuda
         VectorEvaluateAndReduceTestCuda
         VectorBinaryOperationsTestCuda
         VectorUnaryOperationsTestCuda
         VectorVerticalOperationsTestCuda
         # FIXME: fails due to unspecified launch failure in the CUDA reduction kernel for scalar product,
         #        see https://gitlab.com/tnl-project/tnl/-/issues/82
         #VectorOfStaticVectorsTestCuda
)

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
   if( TNL_BUILD_CUDA )
      add_executable( DistributedArrayTest DistributedArrayTestCuda.cu )
      target_compile_options( DistributedArrayTest PUBLIC ${CUDA_TESTS_FLAGS} )
      target_link_libraries( DistributedArrayTest PUBLIC TNL::TNL_CUDA )
   else()
      add_executable( DistributedArrayTest DistributedArrayTest.cpp )
      target_compile_options( DistributedArrayTest PUBLIC ${CXX_TESTS_FLAGS} )
      target_link_libraries( DistributedArrayTest PUBLIC TNL::TNL_CXX )
   endif()
   target_link_libraries( DistributedArrayTest PUBLIC ${TESTS_LIBRARIES} )
   target_link_options( DistributedArrayTest PUBLIC ${TESTS_LINKER_FLAGS} )

   ADD_EXECUTABLE( DistributedVectorBinaryOperationsTest DistributedVectorBinaryOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedVectorBinaryOperationsTest PUBLIC ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorBinaryOperationsTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( DistributedVectorBinaryOperationsTest PUBLIC ${TESTS_LINKER_FLAGS} )

   ADD_EXECUTABLE( DistributedVectorUnaryOperationsTest DistributedVectorUnaryOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedVectorUnaryOperationsTest PUBLIC ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorUnaryOperationsTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( DistributedVectorUnaryOperationsTest PUBLIC ${TESTS_LINKER_FLAGS} )

   ADD_EXECUTABLE( DistributedVectorVerticalOperationsTest DistributedVectorVerticalOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedVectorVerticalOperationsTest PUBLIC ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorVerticalOperationsTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( DistributedVectorVerticalOperationsTest PUBLIC ${TESTS_LINKER_FLAGS} )

   foreach( target IN ITEMS DistributedArrayTest DistributedVectorBinaryOperationsTest DistributedVectorUnaryOperationsTest DistributedVectorVerticalOperationsTest )
      # enable MPI support in TNL
      target_compile_definitions( ${target} PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( ${target} PUBLIC MPI::MPI_CXX )
   endforeach()

   if( TNL_BUILD_CUDA )
      add_executable( DistributedVectorBinaryOperationsTestCuda DistributedVectorBinaryOperationsTestCuda.cu )
      target_compile_options( DistributedVectorBinaryOperationsTestCuda PUBLIC ${CUDA_TESTS_FLAGS} )
      target_link_libraries( DistributedVectorBinaryOperationsTestCuda PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
      target_link_options( DistributedVectorBinaryOperationsTestCuda PUBLIC ${TESTS_LINKER_FLAGS} )

      add_executable( DistributedVectorUnaryOperationsTestCuda DistributedVectorUnaryOperationsTestCuda.cu )
      target_compile_options( DistributedVectorUnaryOperationsTestCuda PUBLIC ${CUDA_TESTS_FLAGS} )
      target_link_libraries( DistributedVectorUnaryOperationsTestCuda PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
      target_link_options( DistributedVectorUnaryOperationsTestCuda PUBLIC ${TESTS_LINKER_FLAGS} )

      add_executable( DistributedVectorVerticalOperationsTestCuda DistributedVectorVerticalOperationsTestCuda.cu )
      target_compile_options( DistributedVectorVerticalOperationsTestCuda PUBLIC ${CUDA_TESTS_FLAGS} )
      target_link_libraries( DistributedVectorVerticalOperationsTestCuda PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
      target_link_options( DistributedVectorVerticalOperationsTestCuda PUBLIC ${TESTS_LINKER_FLAGS} )

      foreach( target IN ITEMS DistributedVectorBinaryOperationsTestCuda DistributedVectorUnaryOperationsTestCuda DistributedVectorVerticalOperationsTestCuda )
         # enable MPI support in TNL
         target_compile_definitions( ${target} PUBLIC "-DHAVE_MPI" )
         # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
         target_link_libraries( ${target} PUBLIC MPI::MPI_CXX )
      endforeach()
   endif()

   add_test_mpi( NAME DistributedArrayTest NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedArrayTest${CMAKE_EXECUTABLE_SUFFIX}" )
   add_test_mpi( NAME DistributedArrayTest_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedArrayTest${CMAKE_EXECUTABLE_SUFFIX}" )

   add_test_mpi( NAME DistributedVectorBinaryOperationsTest NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorBinaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )
   add_test_mpi( NAME DistributedVectorBinaryOperationsTest_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorBinaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )

   add_test_mpi( NAME DistributedVectorUnaryOperationsTest NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorUnaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )
   add_test_mpi( NAME DistributedVectorUnaryOperationsTest_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorUnaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )

   add_test_mpi( NAME DistributedVectorVerticalOperationsTest NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorVerticalOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )
   add_test_mpi( NAME DistributedVectorVerticalOperationsTest_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorVerticalOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )

   if( TNL_BUILD_CUDA )
      add_test_mpi( NAME DistributedVectorBinaryOperationsTestCuda NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorBinaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      add_test_mpi( NAME DistributedVectorBinaryOperationsTestCuda_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorBinaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )

      add_test_mpi( NAME DistributedVectorUnaryOperationsTestCuda NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorUnaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      add_test_mpi( NAME DistributedVectorUnaryOperationsTestCuda_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorUnaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )

      add_test_mpi( NAME DistributedVectorVerticalOperationsTestCuda NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorVerticalOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      add_test_mpi( NAME DistributedVectorVerticalOperationsTestCuda_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorVerticalOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
   endif()
endif()
