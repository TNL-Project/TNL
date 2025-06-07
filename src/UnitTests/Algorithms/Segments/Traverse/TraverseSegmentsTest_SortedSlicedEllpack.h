#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SortedSlicedEllpackTraverseSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SortedSlicedEllpackSegmentsTypes = ::testing::Types<
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, long > >,
   TNL::Algorithms::Segments::SortedSegments<
      TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Host,
                                                int,
                                                TNL::Allocators::Default< TNL::Devices::Host >::template Allocator< int >,
                                                TNL::Algorithms::Segments::ColumnMajorOrder > >,
   TNL::Algorithms::Segments::SortedSegments<
      TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Host,
                                                long,
                                                TNL::Allocators::Default< TNL::Devices::Host >::template Allocator< long >,
                                                TNL::Algorithms::Segments::ColumnMajorOrder > >
#if defined( __CUDACC__ )
   ,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, long > >,
   TNL::Algorithms::Segments::SortedSegments<
      TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Cuda,
                                                int,
                                                TNL::Allocators::Default< TNL::Devices::Cuda >::template Allocator< int >,
                                                TNL::Algorithms::Segments::ColumnMajorOrder > >,
   TNL::Algorithms::Segments::SortedSegments<
      TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Cuda,
                                                long,
                                                TNL::Allocators::Default< TNL::Devices::Cuda >::template Allocator< long >,
                                                TNL::Algorithms::Segments::ColumnMajorOrder > >
#elif defined( __HIP__ )
   ,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, long > >,
   TNL::Algorithms::Segments::SortedSegments<
      TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Hip,
                                                int,
                                                TNL::Allocators::Default< TNL::Devices::Hip >::template Allocator< int >,
                                                TNL::Algorithms::Segments::ColumnMajorOrder > >,
   TNL::Algorithms::Segments::SortedSegments<
      TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Hip,
                                                long,
                                                TNL::Allocators::Default< TNL::Devices::Hip >::template Allocator< long >,
                                                TNL::Algorithms::Segments::ColumnMajorOrder > >
#endif
   >;

TYPED_TEST_SUITE( SortedSlicedEllpackTraverseSegmentsTest, SortedSlicedEllpackSegmentsTypes );

TYPED_TEST( SortedSlicedEllpackTraverseSegmentsTest, forElements_EmptySegments )
{
   test_forElements_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackTraverseSegmentsTest, forElements_EqualSizes )
{
   test_forElements_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackTraverseSegmentsTest, forElements )
{
   test_forElements< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackTraverseSegmentsTest, forElementsIf )
{
   test_forElementsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   test_forElementsWithSegmentIndexes_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   test_forElementsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackTraverseSegmentsTest, forSegments )
{
   test_forSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackTraverseSegmentsTest, forSegmentsWithIndexes )
{
   test_forSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackTraverseSegmentsTest, forSegmentsIf )
{
   test_forSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackTraverseSegmentsTest, forSegmentsSequential )
{
   test_forSegmentsSequential< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
