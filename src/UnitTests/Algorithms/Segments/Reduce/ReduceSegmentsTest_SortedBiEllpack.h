#pragma once

#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>
#include <TNL/Algorithms/Segments/SortedSegmentsView.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SortedBiEllpackReduceSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SortedBiEllpackSegmentsTypes = ::testing::Types<
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Host, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Host, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Host, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Host, long > >
#if defined( __CUDACC__ )
   ,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Cuda, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Cuda, long > >
#elif defined( __HIP__ )
   ,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Hip, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Hip, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Hip, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Hip, long > >
#endif
   >;

TYPED_TEST_SUITE( SortedBiEllpackReduceSegmentsTest, SortedBiEllpackSegmentsTypes );

TYPED_TEST( SortedBiEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   test_reduceSegments_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   test_reduceSegments_MaximumInSegments_short_fetch< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   test_reduceSegmentsWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   test_reduceSegmentsIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackReduceSegmentsTest, reduceSegmentsIfWithArgument_MaximumInSegments )
{
   test_reduceSegmentsIfWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
