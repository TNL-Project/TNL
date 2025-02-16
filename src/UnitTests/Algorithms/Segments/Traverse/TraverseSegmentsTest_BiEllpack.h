#include <TNL/Algorithms/Segments/BiEllpack.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class BiEllpackTraverseSegmentsTest : public ::testing::Test
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
TYPED_TEST_SUITE( BiEllpackTraverseSegmentsTest, BiEllpackSegmentsTypes );

TYPED_TEST( BiEllpackTraverseSegmentsTest, forElements_EmptySegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElements_EmptySegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackTraverseSegmentsTest, forElements_EqualSizes )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElements_EqualSizes< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackTraverseSegmentsTest, forElements )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElements< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackTraverseSegmentsTest, forElementsIf )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElementsIf< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElementsWithSegmentIndexes_EmptySegments< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forElementsWithSegmentIndexes< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackTraverseSegmentsTest, forSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_forSegments< BiEllpackSegmentsType >();
}

#include "../../../main.h"
