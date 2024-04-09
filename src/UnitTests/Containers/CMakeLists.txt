add_subdirectory(ndarray)

set(CPP_TESTS
    MemoryOperationsTest
    ArrayTest
    ArrayViewTest
    StaticArrayTest
    StaticVectorTest
    StaticVectorOperationsTest
    StaticVectorOfStaticVectorsTest
    VectorTest
    VectorEvaluateAndReduceTest
    VectorBinaryOperationsTest
    VectorUnaryOperationsTest
    VectorVerticalOperationsTest
    VectorOfStaticVectorsTest
    BlockTest
    BlockPartitioningTest
)
set(CUDA_TESTS
    MemoryOperationsTestCuda
    ArrayTestCuda
    ArrayViewTestCuda
    VectorTestCuda
    VectorEvaluateAndReduceTestCuda
    VectorBinaryOperationsTestCuda
    VectorUnaryOperationsTestCuda
    VectorVerticalOperationsTestCuda
    # FIXME: fails due to unspecified launch failure in the CUDA reduction kernel for scalar product,
    #        see https://gitlab.com/tnl-project/tnl/-/issues/82
    #VectorOfStaticVectorsTestCuda
)
set(HIP_TESTS
    MemoryOperationsTestHip
    ArrayTestHip
    ArrayViewTestHip
    VectorTestHip
    VectorBinaryOperationsTestHip
    VectorUnaryOperationsTestHip
    VectorVerticalOperationsTestHip
    VectorOfStaticVectorsTestHip
)

foreach(target IN ITEMS ${CPP_TESTS})
    add_executable(${target} ${target}.cpp)
    target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
    target_link_libraries(${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES})
    target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
    add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
endforeach()

if(TNL_BUILD_CUDA)
    foreach(target IN ITEMS ${CUDA_TESTS})
        add_executable(${target} ${target}.cu)
        target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()

if(TNL_BUILD_HIP)
    foreach(target IN ITEMS ${HIP_TESTS})
        add_executable(${target} ${target}.hip)
        target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL_HIP ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()

set(MPI_TESTS DistributedArrayTest DistributedVectorBinaryOperationsTest DistributedVectorUnaryOperationsTest
              DistributedVectorVerticalOperationsTest
)

if(TNL_BUILD_MPI)
    foreach(target IN ITEMS ${MPI_TESTS})
        add_executable(${target} ${target}.cpp)
        target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})

        # enable MPI support in TNL
        target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)

        add_test_mpi(NAME ${target} NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
        add_test_mpi(
            NAME ${target}_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
        )
    endforeach()

    if(TNL_BUILD_CUDA)
        foreach(test IN ITEMS ${MPI_TESTS})
            set(target ${test}Cuda)
            add_executable(${target} ${target}.cu)
            target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
            target_link_libraries(${target} PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES})
            target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})

            # enable MPI support in TNL
            target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
            # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
            target_link_libraries(${target} PUBLIC MPI::MPI_CXX)

            add_test_mpi(NAME ${target} NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
            add_test_mpi(
                NAME ${target}_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
            )
        endforeach()
    endif()

    if(TNL_BUILD_HIP)
        foreach(test IN ITEMS ${MPI_TESTS})
            set(target ${test}Hip)
            add_executable(${target} ${target}.hip)
            target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
            target_link_libraries(${target} PUBLIC TNL::TNL_HIP ${TESTS_LIBRARIES})
            target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})

            # enable MPI support in TNL
            target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
            # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
            target_link_libraries(${target} PUBLIC MPI::MPI_CXX)

            add_test_mpi(NAME ${target} NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
            add_test_mpi(
                NAME ${target}_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
            )
        endforeach()
    endif()
endif()
