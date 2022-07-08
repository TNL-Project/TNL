set( CPP_TESTS  NDSubarrayTest SlicedNDArrayTest StaticNDArrayTest )
set( CUDA_TESTS  StaticNDArrayCudaTest )
if( BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} NDArrayTest )
else()
   set( CPP_TESTS  ${CPP_TESTS} NDArrayTest )
endif()

foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PRIVATE ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} ${TESTS_LIBRARIES} )
   target_link_options( ${target} PRIVATE ${TESTS_LINKER_FLAGS} )
   add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
endforeach()

if( BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_TESTS} )
      cuda_add_executable( ${target} ${target}.cu OPTIONS ${CUDA_TESTS_FLAGS} )
      target_link_libraries( ${target} ${TESTS_LIBRARIES} )
      target_link_options( ${target} PRIVATE ${TESTS_LINKER_FLAGS} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
endif()

if( ${BUILD_MPI} )
   set( MPI_TESTS
            DistributedNDArray_1D_test
            DistributedNDArray_semi1D_test
            DistributedNDArrayOverlaps_1D_test
            DistributedNDArrayOverlaps_semi1D_test
   )
   if( BUILD_CUDA )
      foreach( target IN ITEMS ${MPI_TESTS} )
         cuda_add_executable( ${target} ${target}.cu OPTIONS ${CUDA_TESTS_FLAGS} )
      endforeach()
   else()
      foreach( target IN ITEMS ${MPI_TESTS} )
         add_executable( ${target} ${target}.cpp )
         target_compile_options( ${target} PRIVATE ${CXX_TESTS_FLAGS} )
      endforeach()
   endif()

   foreach( target IN ITEMS ${MPI_TESTS} )
      target_link_libraries( ${target} ${TESTS_LIBRARIES} )
      target_link_options( ${target} PRIVATE ${TESTS_LINKER_FLAGS} )

      # enable MPI support in TNL
      target_compile_definitions( ${target} PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( ${target} MPI::MPI_CXX )

      set( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}" )
      add_test( NAME ${target} COMMAND "mpirun" ${mpi_test_parameters})
      add_test( NAME ${target}_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}" )
   endforeach()
endif()
