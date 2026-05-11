add_subdirectory(Reduce)
add_subdirectory(Traverse)

set(CPP_TESTS
    SegmentsTest_AdaptiveCSR
    SegmentsTest_BiEllpack
    SegmentsTest_ChunkedEllpack
    SegmentsTest_CSR
    SegmentsTest_Ellpack
    SegmentsTest_SlicedEllpack
    SegmentsTest_SortedAdaptiveCSR
    SegmentsTest_SortedBiEllpack
    SegmentsTest_SortedChunkedEllpack
    SegmentsTest_SortedCSR
    SegmentsTest_SortedEllpack
    SegmentsTest_SortedSlicedEllpack
    GrowingSegmentsTest
)

set(CUDA_TESTS
    SegmentsTest_AdaptiveCSR_cuda
    SegmentsTest_BiEllpack_cuda
    SegmentsTest_ChunkedEllpack_cuda
    SegmentsTest_CSR_cuda
    SegmentsTest_Ellpack_cuda
    SegmentsTest_SlicedEllpack_cuda
    SegmentsTest_SortedAdaptiveCSR_cuda
    SegmentsTest_SortedBiEllpack_cuda
    SegmentsTest_SortedChunkedEllpack_cuda
    SegmentsTest_SortedCSR_cuda
    SegmentsTest_SortedEllpack_cuda
    SegmentsTest_SortedSlicedEllpack_cuda
    GrowingSegmentsTest_cuda
)

set(HIP_TESTS
    SegmentsTest_AdaptiveCSR_hip
    SegmentsTest_BiEllpack_hip
    SegmentsTest_ChunkedEllpack_hip
    SegmentsTest_CSR_hip
    SegmentsTest_Ellpack_hip
    SegmentsTest_SlicedEllpack_hip
    SegmentsTest_SortedAdaptiveCSR_hip
    SegmentsTest_SortedBiEllpack_hip
    SegmentsTest_SortedChunkedEllpack_hip
    SegmentsTest_SortedCSR_hip
    SegmentsTest_SortedEllpack_hip
    SegmentsTest_SortedSlicedEllpack_hip
    GrowingSegmentsTest_hip
)

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
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()
