#include <TNL/Algorithms/Segments/SlicedEllpack.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SlicedEllpackTraverseSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SlicedEllpackSegmentsTypes =
   ::testing::Types< typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, long >::BaseType
#if defined( __CUDACC__ )
                     ,
                     typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Cuda, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Cuda, long >::BaseType
#elif defined( __HIP__ )
                     ,
                     typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Hip, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Hip, long >::BaseType
#endif
                     >;

TYPED_TEST_SUITE( SlicedEllpackTraverseSegmentsTest, SlicedEllpackSegmentsTypes );

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElements_EmptySegments )
{
   test_forElements_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElements_EqualSizes )
{
   test_forElements_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElements )
{
   test_forElements< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElementsIf )
{
   test_forElementsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   test_forElementsWithSegmentIndexes_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   test_forElementsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forSegments )
{
   test_forSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forSegmentsWithIndexes )
{
   test_forSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forSegmentsIf )
{
   test_forSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forSegmentsSequential )
{
   test_forSegmentsSequential< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
