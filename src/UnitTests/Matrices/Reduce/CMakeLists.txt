set(TESTS DenseMatrixReduceTest LambdaMatrixReduceTest MultidiagonalMatrixReduceTest TridiagonalMatrixReduceTest)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h LABELS Matrices)
endforeach()

# generates the SparseMatrixReduceTest executables for all segments types in
# SPARSE_MATRIX_TESTED_SEGMENTS_TYPES (see ../CMakeLists.txt)
tnl_add_segments_tests(
    SparseMatrixReduceTest
    SUITE MatrixReduceTest
    PREFIX SparseMatrix<SEGMENTS_TYPE>
    TYPES "${SPARSE_MATRIX_TEST_TYPES}"
    GPU_TYPES "${SPARSE_MATRIX_TEST_GPU_TYPES}"
    HEADERS ../SparseMatrixTestTypes.h MatrixReduceTest.h
    SEGMENTS_TYPES ${SPARSE_MATRIX_TESTED_SEGMENTS_TYPES}
    LABELS Matrices
)
