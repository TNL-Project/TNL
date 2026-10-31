foreach(test IN ITEMS NDSubarrayTest SyncDirectionsTest)
    tnl_add_test_executables(${test} SOURCES ${test}.cpp BACKENDS HOST)
endforeach()

# test with different sources for the host and GPU backends; the CUDA source is also used for HIP
tnl_add_test_executables(
    StaticNDArrayTest
    SOURCES StaticNDArrayTest.cpp
    CUDA_SOURCES StaticNDArrayTestCuda.cu
    HIP_SOURCES StaticNDArrayTestCuda.cu
)

# tests built only for one backend (GPU if available)
set(COMMON_TESTS
    NDArrayTest
    NDArrayReduce_1D_test
    NDArrayReduce_2D_test
    NDArrayReduce_3D_test
    NDArrayReduce_4D_test
    NDArrayReduce_5D_test
    NDArrayReduce_6D_test
    NDArrayReduce_7D_test
)

foreach(test IN ITEMS ${COMMON_TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h BACKENDS CUDA HIP HOST SINGLE)
endforeach()

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

foreach(test IN ITEMS ${MPI_TESTS})
    # 4 ranks are enough for 1D tests
    if(${test} MATCHES "1D")
        set(_nproc 4)
    else()
        set(_nproc 12)
    endif()

    tnl_add_test_executables(${test} MPI NPROC ${_nproc} SOURCES ${test}.h BACKENDS CUDA HIP HOST SINGLE)
endforeach()
