set(TESTS
    GraphTest_SparseMatrix
    GraphTest_DenseMatrix
    GraphTest_Traverse_SparseMatrix
    GraphTest_Traverse_DenseMatrix
    GraphTest_Reduce_SparseMatrix
    GraphTest_Reduce_DenseMatrix
)

add_subdirectory(Algorithms)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h)
endforeach()
