set(COMMON_EXAMPLES
    TridiagonalMatrixExample_Constructor_init_list_1
    TridiagonalMatrixExample_setElements
    #   TridiagonalMatrixViewExample_constructor
    TridiagonalMatrixViewExample_getCompressedRowLengths
    #   TridiagonalMatrixViewExample_getElementsCount
    TridiagonalMatrixViewExample_getConstRow
    TridiagonalMatrixViewExample_getRow
    TridiagonalMatrixViewExample_setElement
    TridiagonalMatrixViewExample_addElement
    TridiagonalMatrixViewExample_getElement
    TridiagonalMatrixViewExample_reduceRows
    TridiagonalMatrixViewExample_reduceAllRows
    TridiagonalMatrixViewExample_forElements
    TridiagonalMatrixViewExample_forElementsWithRowIndexes-1
    TridiagonalMatrixViewExample_forElementsWithRowIndexes-2
    TridiagonalMatrixViewExample_forElementsIf
    TridiagonalMatrixViewExample_forAllElements
    TridiagonalMatrixViewExample_forRows
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
