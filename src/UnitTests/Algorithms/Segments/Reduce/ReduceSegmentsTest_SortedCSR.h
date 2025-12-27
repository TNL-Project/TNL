#include <TNL/Algorithms/Segments/SortedSegments.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SortedCSRReduceSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SortedCSRSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Host, int >,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                                 ,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Cuda, int >,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                                 ,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Hip, int >,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Hip, long >
#endif
                                                 >;

TYPED_TEST_SUITE( SortedCSRReduceSegmentsTest, SortedCSRSegmentsTypes );

TYPED_TEST( SortedCSRReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   test_reduceSegments_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRReduceSegmentsTest, reduceSegments_MaximumInTriangularSegments )
{
   test_reduceSegments_MaximumInTriangularSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   test_reduceSegments_MaximumInSegments_short_fetch< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   test_reduceSegmentsWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   test_reduceSegmentsIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRReduceSegmentsTest, reduceSegmentsWithArgumentIf_MaximumInSegments )
{
   test_reduceSegmentsWithArgumentIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRReduceSegmentsTest, reduce_SumOfMaximums )
{
   test_reduce_SumOfMaximums< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRReduceSegmentsTest, reduce_ProductOfSums )
{
   test_reduce_ProductOfSums< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
