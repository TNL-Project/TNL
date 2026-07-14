add_subdirectory(Algorithms)
add_subdirectory(Reduce)
add_subdirectory(Traverse)

# Graph construction and manipulation examples
set(COMMON_EXAMPLES
    GraphExample_Constructors
    GraphExample_setEdges
    GraphExample_VertexView
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
