#include <TNL/Algorithms/Segments/AdaptiveCSR.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class AdaptiveCSRTraverseSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( AdaptiveCSRTraverseSegmentsTest, AdaptiveCSRSegmentsTypes );

TYPED_TEST( AdaptiveCSRTraverseSegmentsTest, forElements_EmptySegments )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_forElements_EmptySegments< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRTraverseSegmentsTest, forElements_EqualSizes )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_forElements_EqualSizes< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRTraverseSegmentsTest, forElements )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_forElements< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRTraverseSegmentsTest, forElementsIf )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_forElementsIf< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_forElementsWithSegmentIndexes_EmptySegments< AdaptiveCSRSegmentsType >();
}

TYPED_TEST( AdaptiveCSRTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   using AdaptiveCSRSegmentsType = typename TestFixture::AdaptiveCSRSegmentsType;

   test_forElementsWithSegmentIndexes< AdaptiveCSRSegmentsType >();
}

#include "../../../main.h"
