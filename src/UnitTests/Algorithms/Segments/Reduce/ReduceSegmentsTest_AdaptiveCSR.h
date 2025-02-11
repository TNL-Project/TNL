#include <TNL/Algorithms/Segments/AdaptiveCSR.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class AdaptiveCSRReduceSegmentsTest : public ::testing::Test
{
protected:
   using AdaptiveCSRSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using AdaptiveCSRSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Host, int >,
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

TYPED_TEST_SUITE( AdaptiveCSRReduceSegmentsTest, AdaptiveCSRSegmentsTypes );

TYPED_TEST( AdaptiveCSRReduceSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;
   test_reduceAllSegments_MaximumInSegments< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRReduceSegmentsTest, reduceAllSegments_MaximumInSegments_short_fetch )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;
   test_reduceAllSegments_MaximumInSegments_short_fetch< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithArgument )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;
   test_reduceAllSegments_MaximumInSegmentsWithArgument< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithSegmentIndexes )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;
   test_reduceAllSegments_MaximumInSegmentsWithSegmentIndexes< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithSegmentIndexesAndArgument )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;
   test_reduceAllSegments_MaximumInSegmentsWithSegmentIndexesAndArgument< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRReduceSegmentsTest, reduceAllSegmentsIf_MaximumInSegments )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;
   test_reduceAllSegmentsIf_MaximumInSegments< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRReduceSegmentsTest, reduceAllSegmentsIfWithArgument_MaximumInSegments )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;
   test_reduceAllSegmentsIfWithArgument_MaximumInSegments< AdaptiveCSRSegmentsType >();
}

#include "../../../main.h"
