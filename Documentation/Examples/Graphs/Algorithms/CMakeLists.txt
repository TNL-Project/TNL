set(COMMON_EXAMPLES
    GraphExample_BFS
    GraphExample_SSSP
    GraphExample_ConnectedComponents
    GraphExample_StronglyConnectedComponents
    GraphExample_Trees
    GraphExample_MaximalIndependentSet
    GraphExample_GraphColoring
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
