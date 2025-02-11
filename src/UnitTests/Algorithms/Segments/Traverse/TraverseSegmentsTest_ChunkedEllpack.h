#include <TNL/Algorithms/Segments/ChunkedEllpack.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class ChunkedEllpackTraverseSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( ChunkedEllpackTraverseSegmentsTest, ChunkedEllpackSegmentsTypes );

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElements_EmptySegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElements_EmptySegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElements_EqualSizes )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElements_EqualSizes< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElements )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElements< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElementsIf )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElementsIf< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElementsWithSegmentIndexes_EmptySegments< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_forElementsWithSegmentIndexes< ChunkedEllpackSegmentsType >();
}

#include "../../../main.h"
