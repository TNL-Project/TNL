set(COMMON_EXAMPLES
    GraphExample_forAllEdges
    GraphExample_forEdges
    GraphExample_forEdgesIf
    GraphExample_forEdgesWithIndexes
    GraphExample_forAllVertices
    GraphExample_forVertices
    GraphExample_forVerticesIf
    GraphExample_forVerticesWithIndexes
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
