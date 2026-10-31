set(COMMON_EXAMPLES
    SparseMatrixExample_Constructor_init_list_1
    SparseMatrixExample_Constructor_init_list_2
    SparseMatrixExample_Constructor_rowCapacities_vector
    SparseMatrixExample_Constructor_std_map
    SparseMatrixExample_getSerializationType
    SparseMatrixExample_setRowCapacities
    SparseMatrixExample_setElements
    SparseMatrixExample_setElements_map
    SparseMatrixExample_getCompressedRowLengths
    SparseMatrixExample_addElement
    SparseMatrixExample_getElement
    SparseMatrixExample_reduceRows
    SparseMatrixExample_reduceAllRows
    SparseMatrixExample_reduceRowsWithFunctional
    SparseMatrixExample_reduceAllRowsWithFunctional
    SparseMatrixExample_forElements
    SparseMatrixExample_forAllElements
    SparseMatrixExample_forElementsIf
    SparseMatrixExample_forElementsWithRowIndexes-1
    SparseMatrixExample_forElementsWithRowIndexes-2
    SparseMatrixExample_forRows
    SparseMatrixViewExample_getConstRow
    SparseMatrixViewExample_getRow
    SparseMatrixViewExample_setElement
    SparseMatrixViewExample_wrapCSR
    SparseMatrixViewExample_wrapEllpack
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
