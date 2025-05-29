#include <TNL/Algorithms/Segments/CSR.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SortedCSRSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SortedCSRSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Host, int >,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                                 ,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Cuda, int >,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                                 ,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Hip, int >,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Hip, long >
#endif
                                                 >;

TYPED_TEST_SUITE( SortedCSRSegmentsTest, SortedCSRSegmentsTypes );

TYPED_TEST( SortedCSRSegmentsTest, isSegments )
{
   test_isSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRSegmentsTest, setSegmentsSizes_EqualSizes )
{
   test_SetSegmentsSizes_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRSegmentsTest, findInSegments )
{
   test_findInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRSegmentsTest, findInSegmentsWithIndexes )
{
   test_findInSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRSegmentsTest, findInSegmentsIf )
{
   test_findInSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRSegmentsTest, sortSegments )
{
   test_sortSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRSegmentsTest, sortSegmentsWithIndexes )
{
   test_sortSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedCSRSegmentsTest, sortSegmentsIf )
{
   test_sortSegmentsIf< typename TestFixture::SegmentsType >();
}

#include "../../main.h"
