add_subdirectory(ndarray)

set(HOST_TESTS
    StaticArrayTest
    StaticVectorTest
    StaticVectorOperationsTest
    StaticVectorOfStaticVectorsTest
    BlockTest
    BlockPartitioningTest
    LinearCombinationTest
)

# tests built for all backends
set(BACKEND_TESTS
    ArrayTest
    ArrayViewTest
    ComplexVectorBinaryOperationsTest
    ComplexVectorUnaryOperationsTest
    ComplexVectorVerticalOperationsTest
    VectorTest
    VectorEvaluateAndReduceTest
    VectorOfStaticVectorsTest
)

# tests built for all backends whose header does not define main()
set(BACKEND_TESTS_WITHOUT_MAIN VectorBinaryOperationsTest VectorUnaryOperationsTest VectorVerticalOperationsTest)

set(MPI_TESTS
    DistributedArrayTest
    DistributedVectorBinaryOperationsTest
    DistributedVectorUnaryOperationsTest
    DistributedVectorVerticalOperationsTest
)

foreach(test IN ITEMS ${HOST_TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.cpp BACKENDS HOST)
endforeach()

foreach(test IN ITEMS ${BACKEND_TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h)
endforeach()

foreach(test IN ITEMS ${BACKEND_TESTS_WITHOUT_MAIN})
    tnl_add_test_executables(${test} SOURCES ${test}.h ../main.h)
endforeach()

# test with different sources for each backend
tnl_add_test_executables(
    MemoryOperationsTest
    SOURCES MemoryOperationsTest.cpp
    CUDA_SOURCES MemoryOperationsTestCuda.cu
    HIP_SOURCES MemoryOperationsTestHip.hip
)

foreach(test IN ITEMS ${MPI_TESTS})
    tnl_add_test_executables(${test} MPI SOURCES ${test}.h)
endforeach()
