add_subdirectory(Accessors)
add_subdirectory(Indexing)
add_subdirectory(Traverse)
add_subdirectory(Neighbours)
add_subdirectory(Utils)

set(COMMON_TESTS)

set(CPU_TESTS
   GridEntitiesOrientationsTest)

set(CUDA_TESTS)

if(BUILD_CUDA)
   set(CUDA_TESTS ${CUDA_TESTS} ${COMMON_TESTS})
else()
   set(CPU_TESTS ${CPU_TESTS} ${COMMON_TESTS})
endif()

foreach(target IN ITEMS ${CPU_TESTS})
   add_executable(${target} ${target}.cpp)
   target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
   target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
   target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
   add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
endforeach()

if(BUILD_CUDA)
   foreach(target IN ITEMS ${CUDA_TESTS})
      add_executable(${target} ${target}.cu)
      target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
      target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
      target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
      add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
   endforeach()
endif()
