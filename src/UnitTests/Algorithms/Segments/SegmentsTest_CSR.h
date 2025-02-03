#include <TNL/Algorithms/Segments/CSR.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class CSRSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( CSRSegmentsTest, CSRSegmentsTypes );
TYPED_TEST( CSRSegmentsTest, setSegmentsSizes_EqualSizes )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_SetSegmentsSizes_EqualSizes< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, forElements_EmptySegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElements_EmptySegments< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, forElements_EqualSizes )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElements_EqualSizes< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, forElements )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElements< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, forElementsIf )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElementsIf< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElementsWithSegmentIndexes_EmptySegments< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, forElementsWithSegmentIndexes )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_forElementsWithSegmentIndexes< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceAllSegments_MaximumInSegments< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, reduceAllSegments_MaximumInSegments_short_fetch )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   test_reduceAllSegments_MaximumInSegments_short_fetch< CSRSegmentsType >();
}

#include "../../main.h"
