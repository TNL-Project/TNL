set(COMMON_EXAMPLES GridExample_constructor-1 GridExample_constructor-2 GridExample_traverse GridExample_writer)

set(CPU_EXAMPLES GridEntitiesOrientationsExample)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.h)
endforeach()

foreach(target IN ITEMS ${CPU_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp BACKENDS HOST)
endforeach()

tnl_add_example_with_output(DistributedGridExample_heatEquation SOURCES DistributedGridExample_heatEquation.h MPI)
