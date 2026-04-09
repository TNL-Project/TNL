set(CPP_TESTS
    ReduceSegmentsTest_AdaptiveCSR
    ReduceSegmentsTest_BiEllpack
    ReduceSegmentsTest_ChunkedEllpack
    ReduceSegmentsTest_CSR
    ReduceSegmentsTest_Ellpack
    ReduceSegmentsTest_SlicedEllpack
    ReduceSegmentsTest_SortedAdaptiveCSR
    ReduceSegmentsTest_SortedBiEllpack
    ReduceSegmentsTest_SortedChunkedEllpack
    ReduceSegmentsTest_SortedCSR
    ReduceSegmentsTest_SortedEllpack
    ReduceSegmentsTest_SortedSlicedEllpack
)

set(CUDA_TESTS
    ReduceSegmentsTest_AdaptiveCSR_cuda
    ReduceSegmentsTest_BiEllpack_cuda
    ReduceSegmentsTest_ChunkedEllpack_cuda
    ReduceSegmentsTest_CSR_cuda
    ReduceSegmentsTest_Ellpack_cuda
    ReduceSegmentsTest_SlicedEllpack_cuda
    ReduceSegmentsTest_SortedAdaptiveCSR_cuda
    ReduceSegmentsTest_SortedBiEllpack_cuda
    ReduceSegmentsTest_SortedChunkedEllpack_cuda
    ReduceSegmentsTest_SortedCSR_cuda
    ReduceSegmentsTest_SortedEllpack_cuda
    ReduceSegmentsTest_SortedSlicedEllpack_cuda
)

set(HIP_TESTS
    ReduceSegmentsTest_AdaptiveCSR_hip
    ReduceSegmentsTest_BiEllpack_hip
    ReduceSegmentsTest_ChunkedEllpack_hip
    ReduceSegmentsTest_CSR_hip
    ReduceSegmentsTest_Ellpack_hip
    ReduceSegmentsTest_SlicedEllpack_hip
    ReduceSegmentsTest_SortedAdaptiveCSR_hip
    ReduceSegmentsTest_SortedBiEllpack_hip
    ReduceSegmentsTest_SortedChunkedEllpack_hip
    ReduceSegmentsTest_SortedCSR_hip
    ReduceSegmentsTest_SortedEllpack_hip
    ReduceSegmentsTest_SortedSlicedEllpack_hip
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
