#include <TNL/Algorithms/Segments/ChunkedEllpack.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class ChunkedEllpackReduceSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using ChunkedEllpackSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Host, int >,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                                      ,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Cuda, int >,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                                      ,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Hip, int >,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Hip, long >
#endif
                                                      >;

TYPED_TEST_SUITE( ChunkedEllpackReduceSegmentsTest, ChunkedEllpackSegmentsTypes );

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   test_reduceSegments_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   test_reduceSegments_MaximumInSegments_short_fetch< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   test_reduceSegmentsWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   test_reduceSegmentsIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegmentsIfWithArgument_MaximumInSegments )
{
   test_reduceSegmentsIfWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
