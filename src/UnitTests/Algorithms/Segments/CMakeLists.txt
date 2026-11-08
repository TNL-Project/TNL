add_subdirectory(Reduce)
add_subdirectory(Traverse)

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
    tnl_add_test_executables(SegmentsTest_${format} SOURCES SegmentsTest_${format}.h)
endforeach()

tnl_add_test_executables(GrowingSegmentsTest SOURCES GrowingSegmentsTest.h)
tnl_add_test_executables(SegmentsLaunchConfigurationTest SOURCES SegmentsLaunchConfigurationTest.h)
