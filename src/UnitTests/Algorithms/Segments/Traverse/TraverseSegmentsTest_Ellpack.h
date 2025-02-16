#include <TNL/Algorithms/Segments/Ellpack.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class EllpackTraverseSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( EllpackTraverseSegmentsTest, EllpackSegmentsTypes );

TYPED_TEST( EllpackTraverseSegmentsTest, forElements_EmptySegments )
{
   test_forElements_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElements_EqualSizes )
{
   test_forElements_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElements )
{
   test_forElements< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElementsIf )
{
   test_forElementsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   test_forElementsWithSegmentIndexes_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   test_forElementsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forSegments )
{
   test_forSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forSegmentsWithIndexes )
{
   test_forSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forSegmentsIf )
{
   test_forSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forSegmentsSequential )
{
   test_forSegmentsSequential< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
