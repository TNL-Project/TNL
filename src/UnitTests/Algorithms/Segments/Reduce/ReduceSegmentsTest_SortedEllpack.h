#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SortedEllpackReduceSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SortedEllpackSegmentsTypes = ::testing::Types<
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Host, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Host, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Host, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Host, long > >
#if defined( __CUDACC__ )
   ,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Cuda, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Cuda, long > >
#elif defined( __HIP__ )
   ,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Hip, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Hip, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Hip, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Hip, long > >
#endif
   >;

TYPED_TEST_SUITE( SortedEllpackReduceSegmentsTest, SortedEllpackSegmentsTypes );

TYPED_TEST( SortedEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   test_reduceSegments_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedEllpackReduceSegmentsTest, reduceSegments_MaximumInTriangularSegments )
{
   test_reduceSegments_MaximumInTriangularSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   test_reduceSegments_MaximumInSegments_short_fetch< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedEllpackReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   test_reduceSegmentsWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedEllpackReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   test_reduceSegmentsIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedEllpackReduceSegmentsTest, reduceSegmentsWithArgumentIf_MaximumInSegments )
{
   test_reduceSegmentsWithArgumentIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedEllpackReduceSegmentsTest, reduce_SumOfMaximums )
{
   test_reduce_SumOfMaximums< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedEllpackReduceSegmentsTest, reduce_ProductOfSums )
{
   test_reduce_ProductOfSums< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
