#pragma once

#include "ReduceSegmentsTest.h"
#include "ReduceSegmentsTest.hpp"

#include "../../../MultipleGrids.h"

// TYPED_TEST_P definitions that will be instantiated for each segment type

TYPED_TEST_P( ReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_reduceSegments_MaximumInSegments< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegments_LongSegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_reduceSegments_LongSegments< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegments_MaximumInTriangularSegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_reduceSegments_MaximumInTriangularSegments< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_reduceSegments_MaximumInSegments_short_fetch< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_reduceSegmentsWithArgument_MaximumInSegments< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_IdentityOfOtherType )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_reduceSegmentsWithSegmentIndexes_IdentityOfOtherType< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_reduceSegmentsIf_MaximumInSegments< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( ReduceSegmentsTest, reduceSegmentsWithArgumentIf_MaximumInSegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_reduceSegmentsWithArgumentIf_MaximumInSegments< typename TestFixture::SegmentsType > );
}

// Register all test cases
REGISTER_TYPED_TEST_SUITE_P(
   ReduceSegmentsTest,
   reduceSegments_MaximumInSegments,
   reduceSegments_LongSegments,
   reduceSegments_MaximumInTriangularSegments,
   reduceSegments_MaximumInSegments_short_fetch,
   reduceSegmentsWithArgument_MaximumInSegments,
   reduceSegmentsWithSegmentIndexes_MaximumInSegments,
   reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments,
   reduceSegmentsWithSegmentIndexes_IdentityOfOtherType,
   reduceSegmentsIf_MaximumInSegments,
   reduceSegmentsWithArgumentIf_MaximumInSegments );
