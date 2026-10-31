add_subdirectory(Segments)
add_subdirectory(Sorting)

# tests built only for one backend (GPU if available)
set(COMMON_TESTS
    compressTest
    containsTest
    findTest
    Reduction2DTest
    Reduction3DTest
)

# tests built for all backends
set(BACKEND_TESTS reduceTest scanTest)

# tests with different sources for each backend: the CUDA and HIP variants
# differ in the device types and in the workarounds for the GPU compilers
set(SEPARATE_TESTS
    copyTest
    equalTest
    fillTest
    parallelForTest
    staticForTest
    unrolledForTest
)

foreach(test IN ITEMS ${COMMON_TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h BACKENDS CUDA HIP HOST SINGLE)
endforeach()

tnl_add_test_executables(AtomicOperationsTest SOURCES AtomicOperationsTest.cpp BACKENDS HOST)
tnl_add_test_executables(SegmentedScanTest SOURCES SegmentedScanTest.h BACKENDS HOST)

foreach(test IN ITEMS ${BACKEND_TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h)
endforeach()

foreach(test IN ITEMS ${SEPARATE_TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.cpp CUDA_SOURCES ${test}Cuda.cu HIP_SOURCES ${test}Hip.hip)
endforeach()

tnl_add_test_executables(distributedScanTest MPI SOURCES distributedScanTest.h)
