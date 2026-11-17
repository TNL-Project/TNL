# formats of segments for which the tests of segments are instantiated (see SegmentsTestFormats.h)
set(SEGMENTS_TEST_FORMATS
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
    "TNL::UnitTests::SegmentsTestTypes< TNL::UnitTests::<FORMAT>Format, TNL::UnitTests::TestDevice, int, long >"
)

add_subdirectory(Reduce)
add_subdirectory(Traverse)

# generates the SegmentsTest executables for all formats of segments in SEGMENTS_TEST_FORMATS
# and adds the Ellpack-specific tests
tnl_add_format_tests(
    SegmentsTest
    SUITE SegmentsTest
    PREFIX <FORMAT>Segments
    TYPES "${SEGMENTS_TEST_TYPES}"
    HEADERS SegmentsTestFormats.h SegmentsTestSuite.hpp
    FORMATS ${SEGMENTS_TEST_FORMATS}
    EXTRA Ellpack:SegmentsTest_Ellpack.h
)

tnl_add_test_executables(GrowingSegmentsTest SOURCES GrowingSegmentsTest.h)
tnl_add_test_executables(SegmentsLaunchConfigurationTest SOURCES SegmentsLaunchConfigurationTest.h)
