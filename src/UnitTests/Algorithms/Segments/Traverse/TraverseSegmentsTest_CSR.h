#include <TNL/Algorithms/Segments/CSR.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class CSRTraverseSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( CSRTraverseSegmentsTest, CSRSegmentsTypes );

TYPED_TEST( CSRTraverseSegmentsTest, forElements_EmptySegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElements_EmptySegments< CSRSegmentsType >();
}

TYPED_TEST( CSRTraverseSegmentsTest, forElements_EqualSizes )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElements_EqualSizes< CSRSegmentsType >();
}

TYPED_TEST( CSRTraverseSegmentsTest, forElements )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElements< CSRSegmentsType >();
}

TYPED_TEST( CSRTraverseSegmentsTest, forElementsIf )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElementsIf< CSRSegmentsType >();
}

TYPED_TEST( CSRTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElementsWithSegmentIndexes_EmptySegments< CSRSegmentsType >();
}

TYPED_TEST( CSRTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElementsWithSegmentIndexes< CSRSegmentsType >();
}

#include "../../../main.h"
