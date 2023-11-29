ADD_SUBDIRECTORY( DistributedMeshes )
ADD_SUBDIRECTORY( Grids )

add_executable( EntityTagsTest EntityTagsTest.cpp )
target_compile_options( EntityTagsTest PUBLIC ${CXX_TESTS_FLAGS} )
target_link_libraries( EntityTagsTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
target_link_options( EntityTagsTest PUBLIC ${TESTS_LINKER_FLAGS} )
add_test( EntityTagsTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/EntityTagsTest${CMAKE_EXECUTABLE_SUFFIX} )

set( COMMON_TESTS
         MeshTest
         MeshIterationTest
         MeshOrderingTest
         MeshGeometryTest
)

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_TESTS} )
      add_executable( ${target} ${target}.cu )
      target_compile_options( ${target} PUBLIC ${CUDA_TESTS_FLAGS} )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
      target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
elseif( TNL_BUILD_HIP )
   foreach( target IN ITEMS ${COMMON_TESTS} )
      add_executable( ${target} ${target}.hip )
      target_compile_options( ${target} PUBLIC ${HIP_TESTS_FLAGS} )
      target_link_libraries( ${target} PUBLIC TNL::TNL_HIP ${TESTS_LIBRARIES} )
      target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
else()
   foreach( target IN ITEMS ${COMMON_TESTS} )
      add_executable( ${target} ${target}.cpp )
      target_compile_options( ${target} PUBLIC ${CXX_TESTS_FLAGS} )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
      target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
endif()

# special tests needing external libraries
set( SPECIAL_TESTS
         NetgenReaderTest
         VTKReaderTest
         VTUReaderTest
         VTIReaderTest
         FPMAReaderTest
)

find_package( ZLIB )
find_package( tinyxml2 )

if( ZLIB_FOUND AND tinyxml2_FOUND )
   foreach( target IN ITEMS ${SPECIAL_TESTS} )
      add_executable(${target} ${target}.cpp)
      target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS} )
      target_link_libraries(${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES})
      target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )

      target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
      target_include_directories(${target} PUBLIC ${ZLIB_INCLUDE_DIRS})
      target_link_libraries(${target} PUBLIC ${ZLIB_LIBRARIES})

      target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
      target_link_libraries(${target} PUBLIC tinyxml2::tinyxml2)

      # configure path to the data directory
      target_compile_definitions(${target} PUBLIC "-DTNL_MESH_TESTS_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")

      add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
   endforeach()
endif()
