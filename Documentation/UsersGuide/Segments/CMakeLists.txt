set(COMMON_EXAMPLES)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()
foreach(target IN ITEMS ${LONG_EXAMPLES})
    tnl_add_backend_executables(${target} SOURCES ${target}.cpp BACKENDS CUDA HOST SINGLE)
endforeach()
