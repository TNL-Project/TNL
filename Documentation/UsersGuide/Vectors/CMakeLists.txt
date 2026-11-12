set(GPU_EXAMPLES Expressions Reduction)
set(HOST_EXAMPLES StaticVectorExample)

foreach(target IN ITEMS ${GPU_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp BACKENDS CUDA HIP)
endforeach()

foreach(target IN ITEMS ${HOST_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp BACKENDS HOST)
endforeach()
