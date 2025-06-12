#include <TNL/Algorithms/Segments/CSR.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class CSRSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
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

TYPED_TEST( CSRSegmentsTest, isSegments )
{
   test_isSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, setSegmentsSizes_EqualSizes )
{
   test_setSegmentsSizes_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, findInSegments )
{
   test_findInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, findInSegmentsWithIndexes )
{
   test_findInSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, findInSegmentsIf )
{
   test_findInSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, sortSegments )
{
   test_sortSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, sortSegmentsWithIndexes )
{
   test_sortSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, sortSegmentsIf )
{
   test_sortSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, scanSegments )
{
   test_scanSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, scanSegmentsWithIndexes )
{
   test_scanSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, scanSegmentsIf )
{
   test_scanSegmentsIf< typename TestFixture::SegmentsType >();
}

#include "../../main.h"
