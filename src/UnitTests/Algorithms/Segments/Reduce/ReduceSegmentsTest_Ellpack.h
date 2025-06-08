#include <TNL/Algorithms/Segments/Ellpack.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class EllpackReduceSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using EllpackSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Host, int >,
                                               TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Host, long >,
                                               TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Host, int >,
                                               TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Host, long >

#if defined( __CUDACC__ )
                                               ,
                                               TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Cuda, int >,
                                               TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Cuda, long >,
                                               TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Cuda, int >,
                                               TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                               ,
                                               TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Hip, int >,
                                               TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Hip, long >,
                                               TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Hip, int >,
                                               TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Hip, long >
#endif
                                               >;

TYPED_TEST_SUITE( EllpackReduceSegmentsTest, EllpackSegmentsTypes );

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   test_reduceSegments_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   test_reduceSegments_MaximumInSegments_short_fetch< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   test_reduceSegmentsWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   test_reduceSegmentsIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegmentsIfWithArg_MaximumInSegments )
{
   test_reduceSegmentsIfWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduce_SumOfMaximums )
{
   test_reduce_SumOfMaximums< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduce_ProductOfSums )
{
   test_reduce_ProductOfSums< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
