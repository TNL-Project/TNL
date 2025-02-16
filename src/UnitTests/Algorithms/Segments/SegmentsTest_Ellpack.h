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

TYPED_TEST( EllpackSegmentsTest, findInSegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_findInSegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, findInSegmentsWithIndexes )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_findInSegmentsWithIndexes< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, findInSegmentsIf )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_findInSegmentsIf< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, sortSegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_sortSegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, sortSegmentsWithSegmentIndexes )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_sortSegmentsWithSegmentIndexes< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, sortSegmentsIf )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_sortSegmentsIf< EllpackSegmentsType >();
}

#include "../../main.h"
