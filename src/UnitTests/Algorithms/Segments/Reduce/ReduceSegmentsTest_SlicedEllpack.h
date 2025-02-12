#include <TNL/Algorithms/Segments/SlicedEllpack.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SlicedEllpackReduceSegmentsTest : public ::testing::Test
{
protected:
   using SlicedEllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SlicedEllpackSegmentsTypes =
   ::testing::Types< typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, long >::BaseType
#if defined( __CUDACC__ )
                     ,
                     typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Cuda, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Cuda, long >::BaseType
#elif defined( __HIP__ )
                     ,
                     typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Hip, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Hip, long >::BaseType
#endif
                     >;

TYPED_TEST_SUITE( SlicedEllpackReduceSegmentsTest, SlicedEllpackSegmentsTypes );

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_reduceSegments_MaximumInSegments< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_reduceSegments_MaximumInSegments_short_fetch< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_reduceSegmentsWithArgument_MaximumInSegments< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_reduceSegmentsIf_MaximumInSegments< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegmentsIfWithArgument_MaximumInSegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_reduceSegmentsIfWithArgument_MaximumInSegments< SlicedEllpackSegmentsType >();
}

#include "../../../main.h"
