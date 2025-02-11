#include <TNL/Algorithms/Segments/Ellpack.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class EllpackSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( EllpackSegmentsTest, EllpackSegmentsTypes );

TYPED_TEST( EllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_SetSegmentsSizes_EqualSizes< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, setSegmentsSizes_EqualSizes_EllpackOnly )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_SetSegmentsSizes_EqualSizes_EllpackOnly< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, forElements_EmptySegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElements_EmptySegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, forElements_EqualSizes )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElements_EqualSizes< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, forElements )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElements< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, forElementsIf )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElementsIf< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElementsWithSegmentIndexes_EmptySegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, forElementsWithSegmentIndexes )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_forElementsWithSegmentIndexes< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, reduceAllSegments_MaximumInSegments_short_fetch )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegments_short_fetch< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, reduceAllSegments_MaximumInSegmentsWithArgument )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegmentsWithArgument< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, reduceAllSegments_MaximumInSegmentsWithSegmentIndexes )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_reduceAllSegments_MaximumInSegmentsWithSegmentIndexes< EllpackSegmentsType >();
}

#include "../../main.h"
