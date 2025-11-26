#pragma once

#include <TNL/Algorithms/Segments/AdaptiveCSR.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SortedAdaptiveCSRReduceSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( SortedAdaptiveCSRReduceSegmentsTest, SortedAdaptiveCSRSegmentsTypes );

TYPED_TEST( SortedAdaptiveCSRReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   test_reduceSegments_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRReduceSegmentsTest, reduceSegments_MaximumInTriangularSegments )
{
   test_reduceSegments_MaximumInTriangularSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   test_reduceSegments_MaximumInSegments_short_fetch< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   test_reduceSegmentsWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   test_reduceSegmentsIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRReduceSegmentsTest, reduceSegmentsIfWithArgument_MaximumInSegments )
{
   test_reduceSegmentsIfWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRReduceSegmentsTest, reduce_SumOfMaximums )
{
   test_reduce_SumOfMaximums< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedAdaptiveCSRReduceSegmentsTest, reduce_ProductOfSums )
{
   test_reduce_ProductOfSums< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
