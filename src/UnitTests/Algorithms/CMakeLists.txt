add_subdirectory(Segments)
add_subdirectory(Sorting)

set(COMMON_TESTS containsTest findTest ParallelForNDTest Reduction2DTest Reduction3DTest)

set(CPP_TESTS
    copyTest
    equalTest
    fillTest
    parallelForTest
    reduceTest
    scanTest
    SegmentedScanTest
    staticForTest
    unrolledForTest
)
set(CUDA_TESTS
    copyTestCuda
    equalTestCuda
    fillTestCuda
    parallelForTestCuda
    reduceTestCuda
    scanTestCuda
    staticForTestCuda
    unrolledForTestCuda
)
set(HIP_TESTS
    copyTestHip
    equalTestHip
    fillTestHip
    parallelForTestHip
    reduceTestHip
    scanTestHip
    staticForTestHip
    unrolledForTestHip
)
if(TNL_BUILD_CUDA)
    set(CUDA_TESTS ${CUDA_TESTS} ${COMMON_TESTS})
elseif(TNL_BUILD_HIP)
    set(HIP_TESTS ${HIP_TESTS} ${COMMON_TESTS})
else()
    set(CPP_TESTS ${CPP_TESTS} ${COMMON_TESTS})
endif()

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

if(TNL_BUILD_MPI)
    add_executable(distributedScanTest distributedScanTest.cpp)
    target_compile_options(distributedScanTest PUBLIC ${CXX_TESTS_FLAGS})
    target_link_libraries(distributedScanTest PUBLIC TNL::TNL ${TESTS_LIBRARIES})
    target_link_options(distributedScanTest PUBLIC ${TESTS_LINKER_FLAGS})
    # enable MPI support in TNL
    target_compile_definitions(distributedScanTest PUBLIC "-DHAVE_MPI")
    # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
    target_link_libraries(distributedScanTest PUBLIC MPI::MPI_CXX)

    if(TNL_BUILD_CUDA)
        add_executable(distributedScanTestCuda distributedScanTestCuda.cu)
        target_compile_options(distributedScanTestCuda PUBLIC ${CUDA_TESTS_FLAGS})
        target_link_libraries(distributedScanTestCuda PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(distributedScanTestCuda PUBLIC ${TESTS_LINKER_FLAGS})
        # enable MPI support in TNL
        target_compile_definitions(distributedScanTestCuda PUBLIC "-DHAVE_MPI")
        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        target_link_libraries(distributedScanTestCuda PUBLIC MPI::MPI_CXX)
    endif()

    if(TNL_BUILD_HIP)
        add_executable(distributedScanTestHip distributedScanTestHip.hip)
        target_compile_options(distributedScanTestHip PUBLIC ${HIP_TESTS_FLAGS})
        target_link_libraries(distributedScanTestHip PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(distributedScanTestHip PUBLIC ${TESTS_LINKER_FLAGS})
        # enable MPI support in TNL
        target_compile_definitions(distributedScanTestHip PUBLIC "-DHAVE_MPI")
        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        target_link_libraries(distributedScanTestHip PUBLIC MPI::MPI_CXX)
    endif()

    add_test_mpi(
        NAME distributedScanTest NPROC 4 COMMAND
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTest${CMAKE_EXECUTABLE_SUFFIX}"
    )
    add_test_mpi(
        NAME distributedScanTest_nodistr NPROC 1 COMMAND
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTest${CMAKE_EXECUTABLE_SUFFIX}"
    )

    if(TNL_BUILD_CUDA)
        add_test_mpi(
            NAME distributedScanTestCuda NPROC 4 COMMAND
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTestCuda${CMAKE_EXECUTABLE_SUFFIX}"
        )
        add_test_mpi(
            NAME distributedScanTestCuda_nodistr NPROC 1 COMMAND
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTestCuda${CMAKE_EXECUTABLE_SUFFIX}"
        )
    endif()
endif()
