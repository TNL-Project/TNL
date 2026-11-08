set(COMMON_EXAMPLES
    MatrixExample_reduceAllRows
    MatrixExample_reduceRows
    MatrixExample_reduceAllRowsWithArgument
    MatrixExample_reduceRowsWithArgument
    MatrixExample_reduceAllRowsIf
    MatrixExample_reduceRowsIf
    MatrixExample_reduceAllRowsWithArgumentIf
    MatrixExample_reduceRowsWithArgumentIf
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
