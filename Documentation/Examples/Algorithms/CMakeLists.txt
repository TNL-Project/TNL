add_subdirectory(Segments)

set(COMMON_EXAMPLES
    compressExample-vector
    compressExample-lambda
    uncompressExample
    SortingExample
    SortingExample2
    SortingExample3
    SequentialForExample
    parallelForScalarExample
    parallelForMultiIndexExample
    reduceArrayExample
    reduceWithArgumentArrayExample
)

set(HOST_EXAMPLES staticForExample unrolledForExample)

set(CUDA_EXAMPLES SortingCUBMergeSort SortingCUBRadixSort)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()

foreach(target IN ITEMS ${CUDA_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cu BACKENDS CUDA)
endforeach()

foreach(target IN ITEMS ${HOST_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp BACKENDS HOST)
endforeach()
