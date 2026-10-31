set(COMMON_EXAMPLES
    ArrayExample
    ArrayExample_forElements
    ArrayViewExample
    ArrayViewExample_forElements
    VectorExample
)

set(MPI_COMMON_EXAMPLES DistributedArrayExample DistributedVectorExample DistributedNDArrayExample)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()

foreach(target IN ITEMS ${MPI_COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp MPI)
endforeach()
