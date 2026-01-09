#pragma once

#include "SegmentsTest.h"
#include "SegmentsTest.hpp"

// TYPED_TEST_P definitions that will be instantiated for each segment type

TYPED_TEST_P( SegmentsTest, isSegments )
{
   test_isSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( SegmentsTest, getView )
{
   test_getView< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( SegmentsTest, setSegmentsSizes_EqualSizes )
{
   test_setSegmentsSizes_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( SegmentsTest, findInSegments )
{
   test_findInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( SegmentsTest, findInSegmentsWithIndexes )
{
   test_findInSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( SegmentsTest, findInSegmentsIf )
{
   test_findInSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( SegmentsTest, sortSegments )
{
   test_sortSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( SegmentsTest, sortSegmentsWithSegmentIndexes )
{
   test_sortSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( SegmentsTest, sortSegmentsIf )
{
   test_sortSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( SegmentsTest, scanSegments )
{
   test_scanSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( SegmentsTest, scanSegmentsWithIndexes )
{
   test_scanSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( SegmentsTest, scanSegmentsIf )
{
   test_scanSegmentsIf< typename TestFixture::SegmentsType >();
}

// Register all test cases
REGISTER_TYPED_TEST_SUITE_P( SegmentsTest,
                             isSegments,
                             getView,
                             setSegmentsSizes_EqualSizes,
                             findInSegments,
                             findInSegmentsWithIndexes,
                             findInSegmentsIf,
                             sortSegments,
                             sortSegmentsWithSegmentIndexes,
                             sortSegmentsIf,
                             scanSegments,
                             scanSegmentsWithIndexes,
                             scanSegmentsIf );
