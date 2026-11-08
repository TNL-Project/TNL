set(COMMON_EXAMPLES
    GraphExample_reduceVertices
    GraphExample_reduceVerticesIf
    GraphExample_reduceVerticesWithArgument
    GraphExample_reduceVerticesWithArgumentIf
    GraphExample_reduceVerticesWithIndexes
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
