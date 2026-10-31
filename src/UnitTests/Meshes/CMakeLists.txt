add_subdirectory(DistributedMeshes)
add_subdirectory(Grids)

tnl_add_test_executables(EntityTagsTest SOURCES EntityTagsTest.h ../main.h BACKENDS HOST)

# tests built only for one backend (GPU if available)
tnl_add_test_executables(MeshIterationTest SOURCES MeshIterationTest.h BACKENDS CUDA HIP HOST SINGLE)
foreach(test IN ITEMS MeshTest MeshOrderingTest MeshGeometryTest)
    tnl_add_test_executables(${test} SOURCES ${test}.h ../main.h BACKENDS CUDA HIP HOST SINGLE)
endforeach()

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
        target_link_libraries(${target} PUBLIC TNL::TNL TNL_tests)

        target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
        target_link_libraries(${target} PUBLIC ZLIB::ZLIB)

        target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
        target_link_libraries(${target} PUBLIC tinyxml2::tinyxml2)

        # configure path to the data directory
        target_compile_definitions(${target} PUBLIC "-DTNL_MESH_TESTS_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")

        tnl_add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()
