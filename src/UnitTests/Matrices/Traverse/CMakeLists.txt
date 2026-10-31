set(TESTS
    DenseMatrixTraverseTest
    LambdaMatrixTraverseTest
    MultidiagonalMatrixTraverseTest
    TridiagonalMatrixTraverseTest
    SparseMatrixTraverseTest_CSR
    SparseMatrixTraverseTest_Ellpack
    SparseMatrixTraverseTest_BiEllpack
    SparseMatrixTraverseTest_ChunkedEllpack
    SparseMatrixTraverseTest_SlicedEllpack
    SparseMatrixTraverseTest_AdaptiveCSR
)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h LABELS Matrices)
endforeach()
