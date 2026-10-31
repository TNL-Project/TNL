set(COMMON_EXAMPLES
    printSegmentsExample-1
    printSegmentsExample-2
    SegmentsExample_constructor_1
    SegmentsExample_constructor_2
    SegmentsExample_getSerializationType
    SegmentsExample_getSegmentsType
    SegmentsExample_setSegmentsSizes
    SegmentsExample_getSegmentView
    SegmentsExample_forElements
    SegmentsExample_forElementsWithIndexes
    SegmentsExample_forElementsIf
    SegmentsExample_forSegments-1
    SegmentsExample_forSegments-2
    SegmentsExample_forSegmentsIf
    SegmentsExample_forSegmentsWithIndexes
    SegmentsExample_reduceSegments
    SegmentsExample_reduceSegmentsIf
    SegmentsExample_reduceSegmentsWithArgumentIf
    SegmentsExample_reduceSegmentsWithSegmentIndexes
    SegmentsExample_reduceSegmentsWithArgument
    SegmentsExample_reduceSegmentsWithSegmentIndexesWithArgument
    SegmentsExample_reduceAll
    SegmentsExample_launchConfiguration
    SegmentsExample_find
    SegmentsExample_scan
    SegmentsExample_sort
    SortedSegmentsExample_constructor
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    if(target STREQUAL "SegmentsExample_forSegmentsIf")
        # this example has different sources for the host and CUDA
        tnl_add_example_with_output(${target} SOURCES ${target}.cpp CUDA_SOURCES ${target}.cu)
    else()
        tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
    endif()
endforeach()
