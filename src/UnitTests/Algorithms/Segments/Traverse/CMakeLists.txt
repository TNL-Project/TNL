set(CPP_TESTS
    TraverseSegmentsTest_AdaptiveCSR
    TraverseSegmentsTest_BiEllpack
    TraverseSegmentsTest_ChunkedEllpack
    TraverseSegmentsTest_CSR
    TraverseSegmentsTest_Ellpack
    TraverseSegmentsTest_SlicedEllpack
    TraverseSegmentsTest_SortedAdaptiveCSR
    TraverseSegmentsTest_SortedBiEllpack
    TraverseSegmentsTest_SortedChunkedEllpack
    TraverseSegmentsTest_SortedCSR
    TraverseSegmentsTest_SortedEllpack
    TraverseSegmentsTest_SortedSlicedEllpack
)

set(CUDA_TESTS
    TraverseSegmentsTest_AdaptiveCSR_cuda
    TraverseSegmentsTest_BiEllpack_cuda
    TraverseSegmentsTest_ChunkedEllpack_cuda
    TraverseSegmentsTest_CSR_cuda
    TraverseSegmentsTest_Ellpack_cuda
    TraverseSegmentsTest_SlicedEllpack_cuda
    TraverseSegmentsTest_SortedAdaptiveCSR_cuda
    TraverseSegmentsTest_SortedBiEllpack_cuda
    TraverseSegmentsTest_SortedChunkedEllpack_cuda
    TraverseSegmentsTest_SortedCSR_cuda
    TraverseSegmentsTest_SortedEllpack_cuda
    TraverseSegmentsTest_SortedSlicedEllpack_cuda
)

set(HIP_TESTS
    TraverseSegmentsTest_AdaptiveCSR_hip
    TraverseSegmentsTest_BiEllpack_hip
    TraverseSegmentsTest_ChunkedEllpack_hip
    TraverseSegmentsTest_CSR_hip
    TraverseSegmentsTest_Ellpack_hip
    TraverseSegmentsTest_SlicedEllpack_hip
    TraverseSegmentsTest_SortedAdaptiveCSR_hip
    TraverseSegmentsTest_SortedBiEllpack_hip
    TraverseSegmentsTest_SortedChunkedEllpack_hip
    TraverseSegmentsTest_SortedCSR_hip
    TraverseSegmentsTest_SortedEllpack_hip
    TraverseSegmentsTest_SortedSlicedEllpack_hip
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
