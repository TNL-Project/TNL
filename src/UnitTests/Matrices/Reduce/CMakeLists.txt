set(COMMON_TESTS
    DenseMatrixReduceTest
    SparseMatrixReduceTest_CSR
    SparseMatrixReduceTest_Ellpack
    SparseMatrixReduceTest_BiEllpack
    SparseMatrixReduceTest_ChunkedEllpack
    SparseMatrixReduceTest_SlicedEllpack
    SparseMatrixReduceTest_AdaptiveCSR
)

set(CPP_TESTS ${COMMON_TESTS})
set(CUDA_TESTS)
set(HIP_TESTS)

foreach(target IN ITEMS ${COMMON_TESTS})
    set(CUDA_TESTS ${CUDA_TESTS} ${target}_cuda)
    set(HIP_TESTS ${HIP_TESTS} ${target}_hip)
endforeach()

foreach(target IN ITEMS ${CPP_TESTS})
    add_executable(${target} ${target}.cpp)
    target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
    target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
    target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
    add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    set_property(TEST ${target} PROPERTY LABELS Matrices)
endforeach()

if(TNL_BUILD_CUDA)
    foreach(target IN ITEMS ${CUDA_TESTS})
        add_executable(${target} ${target}.cu)
        target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
        set_property(TEST ${target} PROPERTY LABELS Matrices)
    endforeach()
endif()

if(TNL_BUILD_HIP)
    foreach(target IN ITEMS ${HIP_TESTS})
        add_executable(${target} ${target}.hip)
        target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
        set_property(TEST ${target} PROPERTY LABELS Matrices)
    endforeach()
endif()
