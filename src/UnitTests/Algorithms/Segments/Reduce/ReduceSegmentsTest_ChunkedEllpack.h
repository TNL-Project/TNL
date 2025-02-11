#include <TNL/Algorithms/Segments/ChunkedEllpack.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class ChunkedEllpackReduceSegmentsTest : public ::testing::Test
{
protected:
   using ChunkedEllpackSegmentsType = Segments;
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

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceAllSegments_MaximumInSegments_short_fetch )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegments_short_fetch< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithArgument )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegmentsWithArgument< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithSegmentIndexes )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegmentsWithSegmentIndexes< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithSegmentIndexesAndArgument )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegmentsWithSegmentIndexesAndArgument< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceAllSegmentsIf_MaximumInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceAllSegmentsIf_MaximumInSegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceAllSegmentsIfWithArgument_MaximumInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceAllSegmentsIfWithArgument_MaximumInSegments< ChunkedEllpackSegmentsType >();
}

#include "../../../main.h"
