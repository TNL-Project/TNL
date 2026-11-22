# generates the TraverseSegmentsTest executables for all segments types in TESTED_SEGMENTS_TYPES
# (see ../CMakeLists.txt)
tnl_add_segments_tests(
    TraverseSegmentsTest
    SUITE TraverseSegmentsTest
    PREFIX <SEGMENTS_TYPE>Segments
    TYPES "${SEGMENTS_TEST_TYPES}"
    HEADERS ../SegmentsTestVariants.h TraverseSegmentsTestSuite.hpp
    SEGMENTS_TYPES ${TESTED_SEGMENTS_TYPES}
)
