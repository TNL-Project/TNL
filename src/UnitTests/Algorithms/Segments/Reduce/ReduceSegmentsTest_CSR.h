#include <TNL/Algorithms/Segments/CSR.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class CSRReduceSegmentsTest : public ::testing::Test
{
protected:
   using CSRSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using CSRSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::CSR< TNL::Devices::Host, int >,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                           ,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, int >,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                           ,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Hip, int >,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Hip, long >
#endif
                                           >;

TYPED_TEST_SUITE( CSRReduceSegmentsTest, CSRSegmentsTypes );

TYPED_TEST( CSRReduceSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceAllSegments_MaximumInSegments< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceAllSegments_MaximumInSegments_short_fetch )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceAllSegments_MaximumInSegments_short_fetch< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithArgument )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceAllSegments_MaximumInSegmentsWithArgument< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithSegmentIndexes )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceAllSegments_MaximumInSegmentsWithSegmentIndexes< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceAllSegments_MaximumInSegmentsWithSegmentIndexesAndArgument )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceAllSegments_MaximumInSegmentsWithSegmentIndexesAndArgument< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceAllSegmentsIf_MaximumInSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceAllSegmentsIf_MaximumInSegments< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceAllSegmentsIfWithArgument_MaximumInSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceAllSegmentsIfWithArgument_MaximumInSegments< CSRSegmentsType >();
}

#include "../../../main.h"
