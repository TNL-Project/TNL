add_subdirectory(DistributedMeshes)
add_subdirectory(Grids)

add_executable(EntityTagsTest EntityTagsTest.cpp)
target_compile_options(EntityTagsTest PUBLIC ${CXX_TESTS_FLAGS})
target_link_libraries(EntityTagsTest PUBLIC TNL::TNL ${TESTS_LIBRARIES})
target_link_options(EntityTagsTest PUBLIC ${TESTS_LINKER_FLAGS})
add_test(EntityTagsTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/EntityTagsTest${CMAKE_EXECUTABLE_SUFFIX})

set(COMMON_TESTS MeshTest MeshIterationTest MeshOrderingTest MeshGeometryTest)

if(TNL_BUILD_CUDA)
    foreach(target IN ITEMS ${COMMON_TESTS})
        add_executable(${target} ${target}.cu)
        target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
elseif(TNL_BUILD_HIP)
    foreach(target IN ITEMS ${COMMON_TESTS})
        add_executable(${target} ${target}.hip)
        target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
else()
    foreach(target IN ITEMS ${COMMON_TESTS})
        add_executable(${target} ${target}.cpp)
        target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()

# special tests needing external libraries
set(SPECIAL_TESTS
    NetgenReaderTest
    VTKReaderTest
    VTUReaderTest
    VTIReaderTest
    FPMAReaderTest
)

if(DEFINED ENV{GITLAB_CI} AND CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    # FIXME: VTIReaderTest fails when compiled with nvc++ in Release mode
    list(REMOVE_ITEM SPECIAL_TESTS VTIReaderTest)
    # FIXME: nvhpc 25.3 has massive memory usage for this test and causes failures in CI
    list(REMOVE_ITEM SPECIAL_TESTS VTUReaderTest)
endif()

find_package(ZLIB)
find_package(tinyxml2)

if(ZLIB_FOUND AND tinyxml2_FOUND)
    foreach(target IN ITEMS ${SPECIAL_TESTS})
        add_executable(${target} ${target}.cpp)
        target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})

        target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
        target_link_libraries(${target} PUBLIC ZLIB::ZLIB)

        target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
        target_link_libraries(${target} PUBLIC tinyxml2::tinyxml2)

        # configure path to the data directory
        target_compile_definitions(${target} PUBLIC "-DTNL_MESH_TESTS_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")

        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()
