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

TYPED_TEST( BiEllpackSegmentsTest, forElements_EmptySegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElements_EmptySegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, forElements_EqualSizes )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElements_EqualSizes< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, forElements )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElements< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, forElementsIf )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElementsIf< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElementsWithSegmentIndexes_EmptySegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, forElementsWithSegmentIndexes )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElementsWithSegmentIndexes< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, reduceAllSegments_MaximumInSegments_short_fetch )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegments_short_fetch< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, reduceAllSegments_MaximumInSegmentsWithArgument )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegmentsWithArgument< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, reduceAllSegments_MaximumInSegmentsWithSegmentIndexes )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegmentsWithSegmentIndexes< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, reduceAllSegments_MaximumInSegmentsWithSegmentIndexesAndArgument )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegmentsWithSegmentIndexesAndArgument< BiEllpackSegmentsType >();
}

#include "../../main.h"
