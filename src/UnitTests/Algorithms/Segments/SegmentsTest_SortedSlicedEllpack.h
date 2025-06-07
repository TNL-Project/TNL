#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SortedSlicedEllpackSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( SortedSlicedEllpackSegmentsTest, SortedSlicedEllpackSegmentsTypes );

TYPED_TEST( SortedSlicedEllpackSegmentsTest, isSegments )
{
   test_isSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   test_SetSegmentsSizes_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackSegmentsTest, findInSegments )
{
   test_findInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackSegmentsTest, findInSegmentsWithIndexes )
{
   test_findInSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackSegmentsTest, findInSegmentsIf )
{
   test_findInSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackSegmentsTest, sortSegments )
{
   test_sortSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackSegmentsTest, sortSegmentsWithSegmentIndexes )
{
   test_sortSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedSlicedEllpackSegmentsTest, sortSegmentsIf )
{
   test_sortSegmentsIf< typename TestFixture::SegmentsType >();
}

#include "../../main.h"
