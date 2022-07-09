set( CPP_TESTS MPICommTest MPIUtilsTest )

if( ${BUILD_MPI} )
foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PRIVATE ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} ${TESTS_LIBRARIES} )
   target_link_options( ${target} PRIVATE ${TESTS_LINKER_FLAGS} )

   # enable MPI support in TNL
   target_compile_definitions( ${target} PUBLIC "-DHAVE_MPI" )
   # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
   target_link_libraries( ${target} MPI::MPI_CXX )

   foreach( np IN ITEMS 2 3 4 )
      set( mpirun_parameters -np ${np} -H localhost:${np} )
      add_test( NAME ${target}_np${np} COMMAND mpirun ${mpirun_parameters} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
   endforeach()
endforeach()
endif()
