set(CPP_EXAMPLES NDArrayExample_RowMajorArray NDArrayExample_3Dpermutations NDArrayExample_HalfStaticArray)

foreach(target IN ITEMS ${CPP_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp BACKENDS HOST)
endforeach()
