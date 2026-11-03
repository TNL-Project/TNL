#pragma once

#include "TraverseSegmentsTest.h"
#include "TraverseSegmentsTest.hpp"

#include "../../../MultipleGrids.h"

// TYPED_TEST_P definitions that will be instantiated for each segment type

TYPED_TEST_P( TraverseSegmentsTest, forElements_EmptySegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_forElements_EmptySegments< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( TraverseSegmentsTest, forElements_EqualSizes )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_forElements_EqualSizes< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( TraverseSegmentsTest, forElements )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_forElements< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( TraverseSegmentsTest, forElements_LongSegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_forElements_LongSegments< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( TraverseSegmentsTest, forElementsIf )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_forElementsIf< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( TraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_forElementsWithSegmentIndexes_EmptySegments< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( TraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_forElementsWithSegmentIndexes< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( TraverseSegmentsTest, forSegments )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_forSegments< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( TraverseSegmentsTest, forSegmentsWithIndexes )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_forSegmentsWithIndexes< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( TraverseSegmentsTest, forSegmentsIf )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_forSegmentsIf< typename TestFixture::SegmentsType > );
}

TYPED_TEST_P( TraverseSegmentsTest, forSegmentsSequential )
{
   runWithMultipleGrids< typename TestFixture::SegmentsType::DeviceType >(
      test_forSegmentsSequential< typename TestFixture::SegmentsType > );
}

// Register all test cases
REGISTER_TYPED_TEST_SUITE_P(
   TraverseSegmentsTest,
   forElements_EmptySegments,
   forElements_EqualSizes,
   forElements,
   forElements_LongSegments,
   forElementsIf,
   forElementsWithSegmentIndexes_EmptySegments,
   forElementsWithSegmentIndexes,
   forSegments,
   forSegmentsWithIndexes,
   forSegmentsIf,
   forSegmentsSequential );
