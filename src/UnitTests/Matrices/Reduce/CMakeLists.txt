set(TESTS
    DenseMatrixReduceTest
    LambdaMatrixReduceTest
    MultidiagonalMatrixReduceTest
    TridiagonalMatrixReduceTest
    SparseMatrixReduceTest_CSR
    SparseMatrixReduceTest_Ellpack
    SparseMatrixReduceTest_BiEllpack
    SparseMatrixReduceTest_ChunkedEllpack
    SparseMatrixReduceTest_SlicedEllpack
    SparseMatrixReduceTest_AdaptiveCSR
)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h LABELS Matrices)
endforeach()
