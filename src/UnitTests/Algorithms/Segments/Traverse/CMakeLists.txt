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
    tnl_add_test_executables(TraverseSegmentsTest_${format} SOURCES TraverseSegmentsTest_${format}.h)
endforeach()
