set(TESTS DenseMatrixReduceTest LambdaMatrixReduceTest MultidiagonalMatrixReduceTest TridiagonalMatrixReduceTest)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h LABELS Matrices)
endforeach()

# generates the SparseMatrixReduceTest executables for all formats of segments in
# SPARSE_MATRIX_TEST_FORMATS (see ../CMakeLists.txt)
tnl_add_format_tests(
    SparseMatrixReduceTest
    SUITE MatrixReduceTest
    PREFIX SparseMatrix<FORMAT>
    TYPES "${SPARSE_MATRIX_TEST_TYPES}"
    GPU_TYPES "${SPARSE_MATRIX_TEST_GPU_TYPES}"
    HEADERS ../SparseMatrixTestTypes.h MatrixReduceTest.h
    FORMATS ${SPARSE_MATRIX_TEST_FORMATS}
    LABELS Matrices
)
