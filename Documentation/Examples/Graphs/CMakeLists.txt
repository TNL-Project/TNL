add_subdirectory(Algorithms)
add_subdirectory(Reduce)
add_subdirectory(Traverse)

# Graph construction and manipulation examples
set(COMMON_EXAMPLES
    GraphExample_Constructors
    GraphExample_setEdges
    GraphExample_VertexView
    SubGraphExample_BFS
    SubGraphExample_EdgeFilter
    SubGraphExample_MaskedSubGraph
    SubGraphExample_Traverse
    SubGraphExample_VertexFilter
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
