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

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceSegments_MaximumInSegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceSegments_MaximumInSegments_short_fetch< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceSegmentsWithArgument_MaximumInSegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceSegmentsIf_MaximumInSegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackReduceSegmentsTest, reduceSegmentsIfWithArgument_MaximumInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceSegmentsIfWithArgument_MaximumInSegments< ChunkedEllpackSegmentsType >();
}

#include "../../../main.h"
