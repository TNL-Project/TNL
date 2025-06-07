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
