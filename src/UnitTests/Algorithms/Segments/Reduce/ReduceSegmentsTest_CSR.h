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

TYPED_TEST( CSRReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceSegments_MaximumInSegments< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceSegments_MaximumInSegments_short_fetch< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceSegmentsWithArgument_MaximumInSegments< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceSegmentsIf_MaximumInSegments< CSRSegmentsType >();
}

TYPED_TEST( CSRReduceSegmentsTest, reduceSegmentsIfWithArgument_MaximumInSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceSegmentsIfWithArgument_MaximumInSegments< CSRSegmentsType >();
}

#include "../../../main.h"
