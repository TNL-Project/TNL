set(COMMON_TESTS
    DenseMatrixTraverseTest
    SparseMatrixTraverseTest_CSR
    SparseMatrixTraverseTest_Ellpack
    SparseMatrixTraverseTest_BiEllpack
    SparseMatrixTraverseTest_ChunkedEllpack
    SparseMatrixTraverseTest_SlicedEllpack
    SparseMatrixTraverseTest_AdaptiveCSR
)

if(TNL_BUILD_CUDA)
    foreach(target IN ITEMS ${COMMON_TESTS})
        add_executable(${target} ${target}.cu)
        target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
        # add "Matrices" label to the test
        set_property(TEST ${target} PROPERTY LABELS Matrices)
    endforeach()
elseif(TNL_BUILD_HIP)
    foreach(target IN ITEMS ${COMMON_TESTS})
        add_executable(${target} ${target}.hip)
        target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
        # add "Matrices" label to the test
        set_property(TEST ${target} PROPERTY LABELS Matrices)
    endforeach()
else()
    foreach(target IN ITEMS ${COMMON_TESTS})
        add_executable(${target} ${target}.cpp)
        target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
        # add "Matrices" label to the test
        set_property(TEST ${target} PROPERTY LABELS Matrices)
    endforeach()
endif()
