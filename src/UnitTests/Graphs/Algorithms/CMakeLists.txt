set(TESTS
    GraphTest_BFS
    GraphTest_CC
    GraphTest_Coloring
    GraphTest_MIS
    GraphTest_SCC
    GraphTest_SSSP
    GraphTest_trees
    GraphTest_LambdaTraits
)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h)
endforeach()

# TODO: fix and add the CUDA and HIP builds of GraphTest_MST
tnl_add_test_executables(GraphTest_MST SOURCES GraphTest_MST.h BACKENDS HOST)
