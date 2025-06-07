#pragma once

#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>
#include <TNL/Algorithms/Segments/SortedSegmentsView.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SortedChunkedEllpackTraverseSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SortedChunkedEllpackSegmentsTypes = ::testing::Types<
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Host, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Host, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Host, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Host, long > >
#if defined( __CUDACC__ )
   ,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Cuda, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Cuda, long > >
#elif defined( __HIP__ )
   ,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Hip, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Hip, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Hip, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Hip, long > >
#endif
   >;

TYPED_TEST_SUITE( SortedChunkedEllpackTraverseSegmentsTest, SortedChunkedEllpackSegmentsTypes );

TYPED_TEST( SortedChunkedEllpackTraverseSegmentsTest, forElements_EmptySegments )
{
   test_forElements_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedChunkedEllpackTraverseSegmentsTest, forElements_EqualSizes )
{
   test_forElements_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedChunkedEllpackTraverseSegmentsTest, forElements )
{
   test_forElements< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedChunkedEllpackTraverseSegmentsTest, forElementsIf )
{
   test_forElementsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedChunkedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   test_forElementsWithSegmentIndexes_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedChunkedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   test_forElementsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedChunkedEllpackTraverseSegmentsTest, forSegments )
{
   test_forSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedChunkedEllpackTraverseSegmentsTest, forSegmentsWithIndexes )
{
   test_forSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedChunkedEllpackTraverseSegmentsTest, forSegmentsIf )
{
   test_forSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedChunkedEllpackTraverseSegmentsTest, forSegmentsSequential )
{
   test_forSegmentsSequential< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
