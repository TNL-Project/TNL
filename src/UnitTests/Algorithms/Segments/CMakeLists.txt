# segments types for which the tests of segments are instantiated (see SegmentsTestVariants.h)
set(TESTED_SEGMENTS_TYPES
    CSR
    AdaptiveCSR
    Ellpack
    SlicedEllpack
    ChunkedEllpack
    BiEllpack
    SortedCSR
    SortedAdaptiveCSR
    SortedEllpack
    SortedSlicedEllpack
    SortedChunkedEllpack
    SortedBiEllpack
)
set(SEGMENTS_TEST_TYPES
    "TNL::UnitTests::SegmentsTestTypes< TNL::UnitTests::<SEGMENTS_TYPE>Variants, TNL::UnitTests::TestDevice, int, long >"
)

add_subdirectory(Reduce)
add_subdirectory(Traverse)

# generates the SegmentsTest executables for all segments types in TESTED_SEGMENTS_TYPES
# and adds the Ellpack-specific tests
tnl_add_segments_tests(
    SegmentsTest
    SUITE SegmentsTest
    PREFIX <SEGMENTS_TYPE>Segments
    TYPES "${SEGMENTS_TEST_TYPES}"
    HEADERS SegmentsTestVariants.h SegmentsTestSuite.hpp
    SEGMENTS_TYPES ${TESTED_SEGMENTS_TYPES}
    EXTRA Ellpack:SegmentsTest_Ellpack.h
)

tnl_add_test_executables(GrowingSegmentsTest SOURCES GrowingSegmentsTest.h)
tnl_add_test_executables(SegmentsLaunchConfigurationTest SOURCES SegmentsLaunchConfigurationTest.h)
