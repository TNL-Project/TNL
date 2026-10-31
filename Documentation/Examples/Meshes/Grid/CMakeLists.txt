set(COMMON_EXAMPLES GridExample_constructor-1 GridExample_constructor-2 GridExample_traverse GridExample_writer)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.h)
endforeach()
