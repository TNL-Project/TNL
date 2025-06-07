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

// types for which MatrixTest is instantiated
using SlicedEllpackSegmentsTypes = ::testing::Types<
   TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, int >,
   TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, long >,
   TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Host,
                                             int,
                                             TNL::Allocators::Default< TNL::Devices::Host >::template Allocator< int >,
                                             TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Host,
                                             long,
                                             TNL::Allocators::Default< TNL::Devices::Host >::template Allocator< long >,
                                             TNL::Algorithms::Segments::ColumnMajorOrder >
#if defined( __CUDACC__ )
   ,
   TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, int >,
   TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, long >,
   TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Cuda,
                                             int,
                                             TNL::Allocators::Default< TNL::Devices::Cuda >::template Allocator< int >,
                                             TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Cuda,
                                             long,
                                             TNL::Allocators::Default< TNL::Devices::Cuda >::template Allocator< long >,
                                             TNL::Algorithms::Segments::ColumnMajorOrder >
#elif defined( __HIP__ )
   ,
   TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, int >,
   TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, long >,
   TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Hip,
                                             int,
                                             TNL::Allocators::Default< TNL::Devices::Hip >::template Allocator< int >,
                                             TNL::Algorithms::Segments::ColumnMajorOrder >,
   TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Hip,
                                             long,
                                             TNL::Allocators::Default< TNL::Devices::Hip >::template Allocator< long >,
                                             TNL::Algorithms::Segments::ColumnMajorOrder >
#endif
   >;

TYPED_TEST_SUITE( SlicedEllpackReduceSegmentsTest, SlicedEllpackSegmentsTypes );

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   test_reduceSegments_MaximumInSegments< typename TestFixture::SegmentsType >();
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

#include "../../../main.h"
