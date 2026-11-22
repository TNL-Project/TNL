# generates the ReduceSegmentsTest executables for all segments types in TESTED_SEGMENTS_TYPES
# (see ../CMakeLists.txt) and adds the specific types and tests of AdaptiveCSR and SlicedEllpack
tnl_add_segments_tests(
    ReduceSegmentsTest
    SUITE ReduceSegmentsTest
    PREFIX <SEGMENTS_TYPE>Segments
    TYPES "${SEGMENTS_TEST_TYPES}"
    HEADERS ../SegmentsTestVariants.h ReduceSegmentsTestSuite.hpp
    SEGMENTS_TYPES ${TESTED_SEGMENTS_TYPES}
    TYPES_FOR SlicedEllpack:SlicedEllpackReduceSegmentsTestTypes
    EXTRA AdaptiveCSR:ReduceSegmentsTest_AdaptiveCSR.h SlicedEllpack:ReduceSegmentsTest_SlicedEllpack.h
)
