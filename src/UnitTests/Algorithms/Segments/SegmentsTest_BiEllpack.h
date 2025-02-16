#include <TNL/Algorithms/Segments/BiEllpack.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class BiEllpackSegmentsTest : public ::testing::Test
{
protected:
   using BiEllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using BiEllpackSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Host, int >,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                                 ,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Cuda, int >,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                                 ,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Hip, int >,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Hip, long >
#endif
                                                 >;

TYPED_TEST_SUITE( BiEllpackSegmentsTest, BiEllpackSegmentsTypes );

TYPED_TEST( BiEllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_SetSegmentsSizes_EqualSizes< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, findInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_findInSegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, findInSegmentsWithIndexes )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_findInSegmentsWithIndexes< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, findInSegmentsIf )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_findInSegmentsIf< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, sortSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_sortSegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, sortSegmentsWithSegmentIndexes )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_sortSegmentsWithSegmentIndexes< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, sortSegmentsIf )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_sortSegmentsIf< BiEllpackSegmentsType >();
}

#include "../../main.h"
