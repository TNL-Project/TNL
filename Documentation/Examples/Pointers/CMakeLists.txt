set(COMMON_EXAMPLES UniquePointerExample SharedPointerExample DevicePointerExample)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
