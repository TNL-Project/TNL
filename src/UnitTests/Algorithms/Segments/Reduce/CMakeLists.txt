# generates the ReduceSegmentsTest executables for all formats of segments in SEGMENTS_TEST_FORMATS
# (see ../CMakeLists.txt) and adds the format-specific types and tests of AdaptiveCSR and SlicedEllpack
tnl_add_format_tests(
    NAME ReduceSegmentsTest
    SUITE ReduceSegmentsTest
    PREFIX <FORMAT>Segments
    TYPES "${SEGMENTS_TEST_TYPES}"
    HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../SegmentsTestFormats.h ${CMAKE_CURRENT_SOURCE_DIR}/ReduceSegmentsTestSuite.hpp
    FORMATS ${SEGMENTS_TEST_FORMATS}
    FORMAT_TYPES SlicedEllpack:SlicedEllpackReduceSegmentsTestTypes
    EXTRA
        AdaptiveCSR:${CMAKE_CURRENT_SOURCE_DIR}/ReduceSegmentsTest_AdaptiveCSR.h
        SlicedEllpack:${CMAKE_CURRENT_SOURCE_DIR}/ReduceSegmentsTest_SlicedEllpack.h
    SPLIT
        SlicedEllpack:4
        SortedCSR:2
        SortedAdaptiveCSR:2
        SortedEllpack:4
        SortedSlicedEllpack:4
        SortedChunkedEllpack:4
        SortedBiEllpack:4
)
