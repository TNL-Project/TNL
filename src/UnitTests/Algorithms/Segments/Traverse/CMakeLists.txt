# generates the TraverseSegmentsTest executables for all formats of segments in SEGMENTS_TEST_FORMATS
# (see ../CMakeLists.txt)
tnl_add_format_tests(
    TraverseSegmentsTest
    SUITE TraverseSegmentsTest
    PREFIX <FORMAT>Segments
    TYPES "${SEGMENTS_TEST_TYPES}"
    HEADERS ../SegmentsTestFormats.h TraverseSegmentsTestSuite.hpp
    FORMATS ${SEGMENTS_TEST_FORMATS}
)
