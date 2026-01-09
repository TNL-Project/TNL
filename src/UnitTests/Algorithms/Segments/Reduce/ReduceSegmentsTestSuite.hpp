#pragma once

#include "ReduceSegmentsTest.h"
#include "ReduceSegmentsTest.hpp"

// TYPED_TEST_P definitions that will be instantiated for each segment type

TYPED_TEST_P( ReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   test_reduceSegments_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegments_MaximumInTriangularSegments )
{
   test_reduceSegments_MaximumInTriangularSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   test_reduceSegments_MaximumInSegments_short_fetch< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   test_reduceSegmentsWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   test_reduceSegmentsIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegmentsWithArgumentIf_MaximumInSegments )
{
   test_reduceSegmentsWithArgumentIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

// Register all test cases
REGISTER_TYPED_TEST_SUITE_P( ReduceSegmentsTest,
                             reduceSegments_MaximumInSegments,
                             reduceSegments_MaximumInTriangularSegments,
                             reduceSegments_MaximumInSegments_short_fetch,
                             reduceSegmentsWithArgument_MaximumInSegments,
                             reduceSegmentsWithSegmentIndexes_MaximumInSegments,
                             reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments,
                             reduceSegmentsIf_MaximumInSegments,
                             reduceSegmentsWithArgumentIf_MaximumInSegments );
