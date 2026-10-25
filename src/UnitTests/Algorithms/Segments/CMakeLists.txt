# formats of segments for which the tests of segments are instantiated (see SegmentsTestFormats.h)
set(SEGMENTS_TEST_FORMATS
    CSR
    AdaptiveCSR
    Ellpack
    SlicedEllpack
    ChunkedEllpack
    BiEllpack
    SortedCSR
    SortedAdaptiveCSR
    SortedEllpack
    SortedSlicedEllpack
    SortedChunkedEllpack
    SortedBiEllpack
)
set(SEGMENTS_TEST_TYPES
    "TNL::UnitTests::SegmentsTestTypes< TNL::UnitTests::<FORMAT>Format, TNL::UnitTests::TestDevice, int, long >"
)

add_subdirectory(Reduce)
add_subdirectory(Traverse)

# generates the SegmentsTest executables for all formats of segments in SEGMENTS_TEST_FORMATS
# and adds the Ellpack-specific tests
tnl_add_format_tests(
    NAME SegmentsTest
    SUITE SegmentsTest
    PREFIX <FORMAT>Segments
    TYPES "${SEGMENTS_TEST_TYPES}"
    HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/SegmentsTestFormats.h ${CMAKE_CURRENT_SOURCE_DIR}/SegmentsTestSuite.hpp
    FORMATS ${SEGMENTS_TEST_FORMATS}
    EXTRA Ellpack:${CMAKE_CURRENT_SOURCE_DIR}/SegmentsTest_Ellpack.h
)

set(CPP_TESTS GrowingSegmentsTest)
set(CUDA_TESTS GrowingSegmentsTest_cuda)
set(HIP_TESTS GrowingSegmentsTest_hip)

foreach(target IN ITEMS ${CPP_TESTS})
    add_executable(${target} ${target}.cpp)
    target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
    target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
    target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
    add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
endforeach()

if(TNL_BUILD_CUDA)
    foreach(target IN ITEMS ${CUDA_TESTS})
        add_executable(${target} ${target}.cu)
        target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()

if(TNL_BUILD_HIP)
    foreach(target IN ITEMS ${HIP_TESTS})
        add_executable(${target} ${target}.hip)
        target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        tnl_add_hip_test(NAME ${target} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
    endforeach()
endif()
