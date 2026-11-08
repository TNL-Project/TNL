add_subdirectory(Factorization)
add_subdirectory(Reduce)
add_subdirectory(Traverse)

set(TESTS
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
    MatrixWriterTest
    MatrixWrappingTest
    DenseMatrixMultiplicationTest
    DenseMatrixTranspositionTest
    DenseMatrixTransMultiplication
)

set(HOST_TESTS StaticMatrixTest MatrixTypeTraitsTest MatrixReaderTest)
set(MPI_TESTS DistributedMatrixTest)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h LABELS Matrices)
endforeach()

foreach(test IN ITEMS ${HOST_TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.cpp BACKENDS HOST LABELS Matrices)
endforeach()

foreach(test IN ITEMS ${MPI_TESTS})
    tnl_add_test_executables(${test} MPI SOURCES ${test}.h LABELS Matrices)
endforeach()
