#include <TNL/Algorithms/Segments/Ellpack.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class EllpackTraverseSegmentsTest : public ::testing::Test
{
protected:
   using EllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using EllpackSegmentsTypes =
   ::testing::Types< typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Host, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Host, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Host, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Host, long >::BaseType

#if defined( __CUDACC__ )
                     ,
                     typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Cuda, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Cuda, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Cuda, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Cuda, long >::BaseType
#elif defined( __HIP__ )
                     ,
                     typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Hip, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Hip, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Hip, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Hip, long >::BaseType
#endif
                     >;

TYPED_TEST_SUITE( EllpackTraverseSegmentsTest, EllpackSegmentsTypes );

TYPED_TEST( EllpackTraverseSegmentsTest, forElements_EmptySegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElements_EmptySegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElements_EqualSizes )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElements_EqualSizes< EllpackSegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElements )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElements< EllpackSegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElementsIf )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElementsIf< EllpackSegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElementsWithSegmentIndexes_EmptySegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElementsWithSegmentIndexes< EllpackSegmentsType >();
}

#include "../../../main.h"
