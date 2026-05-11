#pragma once

#include "TraverseSegmentsTest.h"
#include "TraverseSegmentsTest.hpp"

// TYPED_TEST_P definitions that will be instantiated for each segment type

TYPED_TEST_P( TraverseSegmentsTest, forElements_EmptySegments )
{
   test_forElements_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( TraverseSegmentsTest, forElements_EqualSizes )
{
   test_forElements_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( TraverseSegmentsTest, forElements )
{
   test_forElements< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( TraverseSegmentsTest, forElementsIf )
{
   test_forElementsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( TraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   test_forElementsWithSegmentIndexes_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( TraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   test_forElementsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( TraverseSegmentsTest, forSegments )
{
   test_forSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( TraverseSegmentsTest, forSegmentsWithIndexes )
{
   test_forSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( TraverseSegmentsTest, forSegmentsIf )
{
   test_forSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST_P( TraverseSegmentsTest, forSegmentsSequential )
{
   test_forSegmentsSequential< typename TestFixture::SegmentsType >();
}

// Register all test cases
REGISTER_TYPED_TEST_SUITE_P( TraverseSegmentsTest,
                             forElements_EmptySegments,
                             forElements_EqualSizes,
                             forElements,
                             forElementsIf,
                             forElementsWithSegmentIndexes_EmptySegments,
                             forElementsWithSegmentIndexes,
                             forSegments,
                             forSegmentsWithIndexes,
                             forSegmentsIf,
                             forSegmentsSequential );
