# generates the ReduceSegmentsTest executables for all formats of segments in SEGMENTS_TEST_FORMATS
# (see ../CMakeLists.txt) and adds the format-specific types and tests of AdaptiveCSR and SlicedEllpack
tnl_add_format_tests(
    ReduceSegmentsTest
    SUITE ReduceSegmentsTest
    PREFIX <FORMAT>Segments
    TYPES "${SEGMENTS_TEST_TYPES}"
    HEADERS ../SegmentsTestFormats.h ReduceSegmentsTestSuite.hpp
    FORMATS ${SEGMENTS_TEST_FORMATS}
    FORMAT_TYPES SlicedEllpack:SlicedEllpackReduceSegmentsTestTypes
    EXTRA AdaptiveCSR:ReduceSegmentsTest_AdaptiveCSR.h SlicedEllpack:ReduceSegmentsTest_SlicedEllpack.h
)
