#include <TNL/Algorithms/Segments/Ellpack.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class EllpackSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
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
   test_SetSegmentsSizes_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, setSegmentsSizes_EqualSizes_EllpackOnly )
{
   test_SetSegmentsSizes_EqualSizes_EllpackOnly< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, findInSegments )
{
   test_findInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, findInSegmentsWithIndexes )
{
   test_findInSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, findInSegmentsIf )
{
   test_findInSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, sortSegments )
{
   test_sortSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, sortSegmentsWithSegmentIndexes )
{
   test_sortSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, sortSegmentsIf )
{
   test_sortSegmentsIf< typename TestFixture::SegmentsType >();
}

#include "../../main.h"
