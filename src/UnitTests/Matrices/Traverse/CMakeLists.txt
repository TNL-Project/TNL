set(TESTS DenseMatrixTraverseTest LambdaMatrixTraverseTest MultidiagonalMatrixTraverseTest TridiagonalMatrixTraverseTest)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h LABELS Matrices)
endforeach()

# generates the SparseMatrixTraverseTest executables for all segments types in
# SPARSE_MATRIX_TESTED_SEGMENTS_TYPES (see ../CMakeLists.txt)
tnl_add_segments_tests(
    SparseMatrixTraverseTest
    SUITE MatrixTraverseTest
    PREFIX SparseMatrix<SEGMENTS_TYPE>
    TYPES "${SPARSE_MATRIX_TEST_TYPES}"
    GPU_TYPES "${SPARSE_MATRIX_TEST_GPU_TYPES}"
    HEADERS ../SparseMatrixTestTypes.h MatrixTraverseTest.h
    SEGMENTS_TYPES ${SPARSE_MATRIX_TESTED_SEGMENTS_TYPES}
    LABELS Matrices
)
