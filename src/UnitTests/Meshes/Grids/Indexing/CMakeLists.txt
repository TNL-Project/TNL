set( COMMON_TESTS )

set( CPU_TESTS
            Grid1DTest_indexing
            Grid2DTest_indexing
            Grid3DTest_indexing
)

set( CUDA_TESTS )
if( BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} ${COMMON_TESTS} )
else()
   set( CPU_TESTS  ${CPU_TESTS} ${COMMON_TESTS} )
endif()

foreach( target IN ITEMS ${CPU_TESTS} )
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
