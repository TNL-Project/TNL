set(CPP_TESTS NDSubarrayTest SlicedNDArrayTest StaticNDArrayTest SyncDirectionsTest)
set(CUDA_TESTS StaticNDArrayTestCuda)
set(HIP_TESTS StaticNDArrayTestHip)
if(TNL_BUILD_CUDA)
    set(CUDA_TESTS ${CUDA_TESTS} NDArrayTest)
elseif(TNL_BUILD_HIP)
    set(HIP_TESTS ${HIP_TESTS} NDArrayTest)
else()
    set(CPP_TESTS ${CPP_TESTS} NDArrayTest)
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
    set(MPI_TESTS
        DistributedNDArray_1D_test
        DistributedNDArray_semi1D_test
        DistributedNDArrayOverlaps_1D_test
        DistributedNDArrayOverlaps_semi1D_test
        DistributedNDArray_2D_test
        DistributedNDArrayOverlaps_2D_test
        DistributedNDArray_3D_test
        DistributedNDArrayOverlaps_3D_test
    )
    if(TNL_BUILD_CUDA)
        foreach(target IN ITEMS ${MPI_TESTS})
            add_executable(${target} ${target}.cu)
            target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
            target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
            target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        endforeach()
    elseif(TNL_BUILD_HIP)
        foreach(target IN ITEMS ${MPI_TESTS})
            add_executable(${target} ${target}.hip)
            target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
            target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
            target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        endforeach()
    else()
        foreach(target IN ITEMS ${MPI_TESTS})
            add_executable(${target} ${target}.cpp)
            target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
            target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
            target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        endforeach()
    endif()

    foreach(target IN ITEMS ${MPI_TESTS})
        # enable MPI support in TNL
        target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)

        # 4 ranks are enough for 1D tests
        if(${target} MATCHES "1D")
            set(_nproc 4)
        else()
            set(_nproc 12)
        endif()

        add_test_mpi(
            NAME ${target} NPROC ${_nproc} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
        )
        add_test_mpi(
            NAME ${target}_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
        )
    endforeach()
endif()
