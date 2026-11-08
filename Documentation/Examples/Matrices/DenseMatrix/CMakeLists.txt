set(COMMON_EXAMPLES
    DenseMatrixExample_Constructor_init_list
    DenseMatrixExample_Constructor_sparse_init_list
    DenseMatrixExample_Constructor_std_map
    DenseMatrixExample_setElements
    DenseMatrixViewExample_constructor
    DenseMatrixViewExample_getCompressedRowLengths
    DenseMatrixViewExample_getAllocatedElementsCount
    DenseMatrixViewExample_getConstRow
    DenseMatrixViewExample_getRow
    DenseMatrixViewExample_setElement
    DenseMatrixViewExample_addElement
    DenseMatrixViewExample_getElement
    DenseMatrixViewExample_reduceRows
    DenseMatrixViewExample_reduceAllRows
    DenseMatrixViewExample_reduceRowsWithFunctional
    DenseMatrixViewExample_reduceAllRowsWithFunctional
    DenseMatrixViewExample_forElements
    DenseMatrixViewExample_forElementsWithRowIndexes-1
    DenseMatrixViewExample_forElementsWithRowIndexes-2
    DenseMatrixViewExample_forElementsIf
    DenseMatrixViewExample_forRows
    DenseMatrixViewExample_forAllElements
    DenseMatrixViewExample_wrap
    DenseMatrixOperationsExample_getProduct
    DenseMatrixOperationsExample_getTransposition
    DenseMatrixOperationsExample_getInPlaceTransposition
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
