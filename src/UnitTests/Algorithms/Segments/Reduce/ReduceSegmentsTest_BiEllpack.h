#include <TNL/Algorithms/Segments/BiEllpack.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class BiEllpackReduceSegmentsTest : public ::testing::Test
{
protected:
   using BiEllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using BiEllpackSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Host, int >,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                                 ,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Cuda, int >,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                                 ,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Hip, int >,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Hip, long >
#endif
                                                 >;

TYPED_TEST_SUITE( BiEllpackReduceSegmentsTest, BiEllpackSegmentsTypes );

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceAllSegments_MaximumInSegments_short_fetch )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegments_short_fetch< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithArgument )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegmentsWithArgument< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithSegmentIndexes )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegmentsWithSegmentIndexes< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithSegmentIndexesAndArgument )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegmentsWithSegmentIndexesAndArgument< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegmentsIf_MaximumInSegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsIfWithArgument_MaximumInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegmentsIfWithArgument_MaximumInSegments< BiEllpackSegmentsType >();
}

#include "../../../main.h"
