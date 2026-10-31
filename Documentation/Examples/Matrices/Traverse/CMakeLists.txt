set(COMMON_EXAMPLES
    MatrixExample_forElements
    MatrixExample_forAllElements
    MatrixExample_forElementsWithIndexes
    MatrixExample_forElementsIf
    MatrixExample_forAllElementsIf
    MatrixExample_forRows-1
    MatrixExample_forRows-2
    MatrixExample_forAllRows
    MatrixExample_forRowsWithIndexes
    MatrixExample_forRowsIf
    MatrixExample_forAllRowsIf
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
