set(COMMON_EXAMPLES
    LambdaMatrixExample_Constructor
    LambdaMatrixExample_getCompressedRowLengths
    LambdaMatrixExample_getNonzeroElementsCount
    LambdaMatrixExample_Laplace
    LambdaMatrixExample_Laplace_2
    LambdaMatrixExample_getRow
    LambdaMatrixExample_reduceRows
    LambdaMatrixExample_reduceAllRows
    LambdaMatrixExample_forElements
    LambdaMatrixExample_forAllElements
    LambdaMatrixExample_forRows
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
