#include <TNL/Algorithms/Segments/SlicedEllpack.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SlicedEllpackReduceSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

template< typename Device, typename Index, int SliceSize >
using RowMajorSlicedEllpackType = TNL::Algorithms::Segments::
   RowMajorSlicedEllpack< Device, Index, typename TNL::Allocators::Default< Device >::template Allocator< Index >, SliceSize >;

template< typename Device, typename Index, int SliceSize >
using ColumnMajorSlicedEllpackType = TNL::Algorithms::Segments::ColumnMajorSlicedEllpack<
   Device,
   Index,
   typename TNL::Allocators::Default< Device >::template Allocator< Index >,
   SliceSize >;

// types for which MatrixTest is instantiated
using SlicedEllpackSegmentsTypes =
   ::testing::Types< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, int >,
                     TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, long >,
                     TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, int >,
                     TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                     ,
                     RowMajorSlicedEllpackType< TNL::Devices::Cuda, int, 32 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Cuda, int, 8 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Cuda, int, 1 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Cuda, long int, 32 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Cuda, long int, 1 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Cuda, int, 32 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Cuda, int, 8 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Cuda, int, 1 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Cuda, long int, 32 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Cuda, long int, 1 >
#elif defined( __HIP__ )
                     ,
                     RowMajorSlicedEllpackType< TNL::Devices::Hip, int, 32 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Hip, int, 8 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Hip, int, 1 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Hip, long int, 32 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Hip, long int, 1 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Hip, int, 32 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Hip, int, 8 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Hip, int, 1 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Hip, long int, 32 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Hip, long int, 1 >
#endif
                     >;

TYPED_TEST_SUITE( SlicedEllpackReduceSegmentsTest, SlicedEllpackSegmentsTypes );

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   test_reduceSegments_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegments_MaximumInTriangularSegments )
{
   test_reduceSegments_MaximumInTriangularSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   test_reduceSegments_MaximumInSegments_short_fetch< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   test_reduceSegmentsWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   test_reduceSegmentsIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegmentsIfWithArgument_MaximumInSegments )
{
   test_reduceSegmentsIfWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduce_SumOfMaximums )
{
   test_reduce_SumOfMaximums< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduce_ProductOfSums )
{
   test_reduce_ProductOfSums< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
