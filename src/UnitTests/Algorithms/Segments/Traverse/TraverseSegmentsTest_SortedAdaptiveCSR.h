#pragma once

#include <TNL/Algorithms/Segments/AdaptiveCSR.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SortedAdaptiveCSRTraverseSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SortedAdaptiveCSRSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Host, int >,
                                                         TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                                         ,
                                                         TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Cuda, int >,
                                                         TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                                         ,
                                                         TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Hip, int >,
                                                         TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Hip, long >
#endif
                                                         >;

TYPED_TEST_SUITE( SortedAdaptiveCSRTraverseSegmentsTest, SortedAdaptiveCSRSegmentsTypes );

TYPED_TEST( SortedAdaptiveCSRTraverseSegmentsTest, forElements_EmptySegments )
{
   test_forElements_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRTraverseSegmentsTest, forElements_EqualSizes )
{
   test_forElements_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRTraverseSegmentsTest, forElements )
{
   test_forElements< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRTraverseSegmentsTest, forElementsIf )
{
   test_forElementsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   test_forElementsWithSegmentIndexes_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   test_forElementsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRTraverseSegmentsTest, forSegments )
{
   test_forSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRTraverseSegmentsTest, forSegmentsWithIndexes )
{
   test_forSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRTraverseSegmentsTest, forSegmentsIf )
{
   test_forSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRTraverseSegmentsTest, forSegmentsSequential )
{
   test_forSegmentsSequential< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
