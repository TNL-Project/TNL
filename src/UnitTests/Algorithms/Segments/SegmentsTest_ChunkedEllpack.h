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

TYPED_TEST( ChunkedEllpackSegmentsTest, findInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_findInSegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, findInSegmentsWithIndexes )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_findInSegmentsWithIndexes< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, findInSegmentsIf )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_findInSegmentsIf< ChunkedEllpackSegmentsType >();
}

#include "../../main.h"
