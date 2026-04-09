set(CPP_TESTS
    GraphTest_SparseMatrix
    GraphTest_DenseMatrix
    GraphTest_Traverse_SparseMatrix
    GraphTest_Traverse_DenseMatrix
    GraphTest_Reduce_SparseMatrix
    GraphTest_Reduce_DenseMatrix
)

set(CUDA_TESTS
    GraphTest_SparseMatrix_cuda
    GraphTest_DenseMatrix_cuda
    GraphTest_Traverse_SparseMatrix_cuda
    GraphTest_Traverse_DenseMatrix_cuda
    GraphTest_Reduce_SparseMatrix_cuda
    GraphTest_Reduce_DenseMatrix_cuda
)

set(HIP_TESTS
    GraphTest_SparseMatrix_hip
    GraphTest_DenseMatrix_hip
    GraphTest_Traverse_SparseMatrix_hip
    GraphTest_Traverse_DenseMatrix_hip
    GraphTest_Reduce_SparseMatrix_hip
    GraphTest_Reduce_DenseMatrix_hip
)

add_subdirectory(Algorithms)

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
