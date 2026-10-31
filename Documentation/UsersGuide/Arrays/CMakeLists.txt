set(CUDA_EXAMPLES
    ArrayAllocation
    ArrayIO
    ArrayView-1
    ArrayView-2
    ArrayViewForElements
    contains
    ElementsAccessing-1
    ElementsAccessing-2
)
set(HOST_EXAMPLES StaticArrayExample)

foreach(target IN ITEMS ${CUDA_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp BACKENDS CUDA)
endforeach()

foreach(target IN ITEMS ${HOST_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp BACKENDS HOST)
endforeach()
