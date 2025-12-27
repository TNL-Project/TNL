#include <TNL/Algorithms/Segments/BiEllpack.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class BiEllpackReduceSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using BiEllpackSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Host, int >,
                                                 TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Host, long >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Host, int >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                                 ,
                                                 TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Cuda, int >,
                                                 TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Cuda, long >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Cuda, int >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                                 ,
                                                 TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Hip, int >,
                                                 TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Hip, long >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Hip, int >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Hip, long >
#endif
                                                 >;

TYPED_TEST_SUITE( BiEllpackReduceSegmentsTest, BiEllpackSegmentsTypes );

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   test_reduceSegments_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegments_MaximumInTriangularSegments )
{
   test_reduceSegments_MaximumInTriangularSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   test_reduceSegments_MaximumInSegments_short_fetch< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   test_reduceSegmentsWithArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   test_reduceSegmentsIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduceSegmentsWithArgumentIf_MaximumInSegments )
{
   test_reduceSegmentsWithArgumentIf_MaximumInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduce_SumOfMaximums )
{
   test_reduce_SumOfMaximums< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackReduceSegmentsTest, reduce_ProductOfSums )
{
   test_reduce_ProductOfSums< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
