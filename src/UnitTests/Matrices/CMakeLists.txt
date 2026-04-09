add_subdirectory(Factorization)
add_subdirectory(Reduce)
add_subdirectory(Traverse)

set(COMMON_TESTS
    DenseMatrixTest
    DenseMatrixCopyTest
    TridiagonalMatrixTest
    MultidiagonalMatrixTest
    SparseDenseMatrixOperationsTest_CSR
    SparseMatrixOperationsTest_CSR
    SparseMatrixTest_CSR
    SparseMatrixTest_AdaptiveCSR
    SparseMatrixTest_Ellpack
    SparseMatrixTest_SlicedEllpack
    SparseMatrixTest_ChunkedEllpack
    SparseMatrixTest_BiEllpack
    SparseMatrixTest_SortedAdaptiveCSR
    SparseMatrixTest_SortedCSR
    SparseMatrixTest_SortedBiEllpack
    SparseMatrixTest_SortedChunkedEllpack
    SparseMatrixTest_SortedEllpack
    SparseMatrixTest_SortedSlicedEllpack
    SparseMatrixVectorProductTest_CSR
    SparseMatrixVectorProductTest_AdaptiveCSR
    SparseMatrixVectorProductTest_Ellpack
    SparseMatrixVectorProductTest_SlicedEllpack
    SparseMatrixVectorProductTest_ChunkedEllpack
    SparseMatrixVectorProductTest_BiEllpack
    SparseMatrixCopyTest
    BinarySparseMatrixTest_CSR
    BinarySparseMatrixTest_AdaptiveCSR
    BinarySparseMatrixTest_Ellpack
    BinarySparseMatrixTest_SlicedEllpack
    BinarySparseMatrixCopyTest
    SymmetricSparseMatrixTest_CSR
    LambdaMatrixTest
    MatrixOperationsTest
    MatrixWrappingTest
    DenseMatrixMultiplicationTest
    DenseMatrixTranspositionTest
    DenseMatrixTransMultiplication
)

set(CPP_TESTS ${COMMON_TESTS} StaticMatrixTest MatrixTypeTraitsTest)
set(CUDA_TESTS)
set(HIP_TESTS)

foreach(target IN ITEMS ${COMMON_TESTS})
    set(CUDA_TESTS ${CUDA_TESTS} ${target}_cuda)
    set(HIP_TESTS ${HIP_TESTS} ${target}_hip)
endforeach()

set(MPI_TESTS DistributedMatrixTest)
set(MPI_CUDA_TESTS DistributedMatrixTest_cuda)
set(MPI_HIP_TESTS DistributedMatrixTest_hip)

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
        # add "Matrices" label to the test
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
        # add "Matrices" label to the test
        set_property(TEST ${target} PROPERTY LABELS Matrices)
    endforeach()
endif()

if(TNL_BUILD_MPI)
    foreach(target IN ITEMS ${MPI_TESTS})
        add_executable(${target} ${target}.cpp)
        target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
        add_test_mpi(NAME ${target} NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
        add_test_mpi(
            NAME ${target}_nodistr
            NPROC 1
            COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
        )
        # add "Matrices" label to the test
        set_property(TEST ${target} PROPERTY LABELS Matrices)
        set_property(TEST ${target}_nodistr PROPERTY LABELS Matrices)
    endforeach()

    if(TNL_BUILD_CUDA)
        foreach(target IN ITEMS ${MPI_CUDA_TESTS})
            add_executable(${target} ${target}.cu)
            target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
            target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
            target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
            target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
            target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
            add_test_mpi(NAME ${target} NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
            add_test_mpi(
                NAME ${target}_nodistr
                NPROC 1
                COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
            )
            # add "Matrices" label to the test
            set_property(TEST ${target} PROPERTY LABELS Matrices)
            set_property(TEST ${target}_nodistr PROPERTY LABELS Matrices)
        endforeach()
    endif()

    if(TNL_BUILD_HIP)
        foreach(target IN ITEMS ${MPI_HIP_TESTS})
            add_executable(${target} ${target}.hip)
            target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
            target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
            target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
            target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
            target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
            add_test_mpi(NAME ${target} NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
            add_test_mpi(
                NAME ${target}_nodistr
                NPROC 1
                COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
            )
            # add "Matrices" label to the test
            set_property(TEST ${target} PROPERTY LABELS Matrices)
            set_property(TEST ${target}_nodistr PROPERTY LABELS Matrices)
        endforeach()
    endif()
endif()
