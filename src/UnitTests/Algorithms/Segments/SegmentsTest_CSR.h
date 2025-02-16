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

TYPED_TEST( CSRSegmentsTest, findInSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_findInSegments< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, findInSegmentsWithIndexes )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_findInSegmentsWithIndexes< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, findInSegmentsIf )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_findInSegmentsIf< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, sortSegments )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_sortSegments< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, sortSegmentsWithIndexes )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_sortSegmentsWithSegmentIndexes< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, sortSegmentsIf )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_sortSegmentsIf< CSRSegmentsType >();
}

#include "../../main.h"
