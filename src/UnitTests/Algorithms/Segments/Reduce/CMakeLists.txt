set(FORMATS
    AdaptiveCSR
    BiEllpack
    ChunkedEllpack
    CSR
    Ellpack
    SlicedEllpack
    SortedAdaptiveCSR
    SortedBiEllpack
    SortedChunkedEllpack
    SortedCSR
    SortedEllpack
    SortedSlicedEllpack
)

foreach(format IN ITEMS ${FORMATS})
    tnl_add_test_executables(ReduceSegmentsTest_${format} SOURCES ReduceSegmentsTest_${format}.h)
endforeach()
