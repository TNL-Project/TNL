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

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceSegments_MaximumInSegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceSegments_MaximumInSegments_short_fetch< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceSegmentsWithArgument_MaximumInSegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceSegmentsIf_MaximumInSegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsIfWithArgument_MaximumInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceSegmentsIfWithArgument_MaximumInSegments< BiEllpackSegmentsType >();
}

#include "../../../main.h"
