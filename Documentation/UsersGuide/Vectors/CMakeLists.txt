set(CUDA_EXAMPLES Expressions Reduction)
set(HOST_EXAMPLES StaticVectorExample)

foreach(target IN ITEMS ${CUDA_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp BACKENDS CUDA)
endforeach()

foreach(target IN ITEMS ${HOST_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp BACKENDS HOST)
endforeach()
