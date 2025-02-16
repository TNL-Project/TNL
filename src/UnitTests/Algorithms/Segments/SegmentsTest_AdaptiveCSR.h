#include <TNL/Algorithms/Segments/AdaptiveCSR.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class AdaptiveCSRSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( AdaptiveCSRSegmentsTest, AdaptiveCSRSegmentsTypes );

TYPED_TEST( AdaptiveCSRSegmentsTest, setSegmentsSizes_EqualSizes )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_SetSegmentsSizes_EqualSizes< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRSegmentsTest, findInSegments )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_findInSegments< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRSegmentsTest, findInSegmentsWithIndexes )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_findInSegmentsWithIndexes< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRSegmentsTest, findInSegmentsIf )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_findInSegmentsIf< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRSegmentsTest, sortSegments )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_sortSegments< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRSegmentsTest, sortSegmentsWithSegmentIndexes )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_sortSegmentsWithSegmentIndexes< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRSegmentsTest, sortSegmentsIf )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_sortSegmentsIf< AdaptiveCSRSegmentsType >();
}

#include "../../main.h"
