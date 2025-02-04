#include <TNL/Algorithms/Segments/ChunkedEllpack.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class ChunkedEllpackSegmentsTest : public ::testing::Test
{
protected:
   using ChunkedEllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using ChunkedEllpackSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Host, int >,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                                      ,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Cuda, int >,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                                      ,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Hip, int >,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Hip, long >
#endif
                                                      >;

TYPED_TEST_SUITE( ChunkedEllpackSegmentsTest, ChunkedEllpackSegmentsTypes );

TYPED_TEST( ChunkedEllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_SetSegmentsSizes_EqualSizes< ChunkedEllpackSegmentsType >();
}
TYPED_TEST( ChunkedEllpackSegmentsTest, forElements_EmptySegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElements_EmptySegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, forElements_EqualSizes )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElements_EqualSizes< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, forElements )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElements< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, forElementsIf )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElementsIf< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElementsWithSegmentIndexes_EmptySegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, forElementsWithSegmentIndexes )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElementsWithSegmentIndexes< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegments< ChunkedEllpackSegmentsType >();
}

#include "../../main.h"
