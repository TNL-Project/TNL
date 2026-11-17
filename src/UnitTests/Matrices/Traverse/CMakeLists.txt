set(TESTS DenseMatrixTraverseTest LambdaMatrixTraverseTest MultidiagonalMatrixTraverseTest TridiagonalMatrixTraverseTest)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h LABELS Matrices)
endforeach()

# generates the SparseMatrixTraverseTest executables for all formats of segments in
# SPARSE_MATRIX_TEST_FORMATS (see ../CMakeLists.txt)
tnl_add_format_tests(
    SparseMatrixTraverseTest
    SUITE MatrixTraverseTest
    PREFIX SparseMatrix<FORMAT>
    TYPES "${SPARSE_MATRIX_TEST_TYPES}"
    GPU_TYPES "${SPARSE_MATRIX_TEST_GPU_TYPES}"
    HEADERS ../SparseMatrixTestTypes.h MatrixTraverseTest.h
    FORMATS ${SPARSE_MATRIX_TEST_FORMATS}
    LABELS Matrices
)
