set(COMMON_EXAMPLES
    MultidiagonalMatrixExample_Constructor
    MultidiagonalMatrixExample_Constructor_init_list_1
    MultidiagonalMatrixExample_Constructor_init_list_2
    MultidiagonalMatrixExample_setElements
    #    MultidiagonalMatrixViewExample_constructor
    MultidiagonalMatrixViewExample_getCompressedRowLengths
    #    MultidiagonalMatrixViewExample_getElementsCount
    MultidiagonalMatrixViewExample_getConstRow
    MultidiagonalMatrixViewExample_getRow
    MultidiagonalMatrixViewExample_setElement
    MultidiagonalMatrixViewExample_addElement
    MultidiagonalMatrixViewExample_getElement
    MultidiagonalMatrixViewExample_reduceRows
    MultidiagonalMatrixViewExample_reduceAllRows
    MultidiagonalMatrixViewExample_forElements
    MultidiagonalMatrixViewExample_forElementsWithRowIndexes-1
    MultidiagonalMatrixViewExample_forElementsWithRowIndexes-2
    MultidiagonalMatrixViewExample_forElementsIf
    MultidiagonalMatrixViewExample_forAllElements
    MultidiagonalMatrixViewExample_forRows
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
