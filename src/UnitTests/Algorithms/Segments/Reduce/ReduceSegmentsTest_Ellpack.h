#include <TNL/Algorithms/Segments/Ellpack.h>

#include "ReduceSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class EllpackReduceSegmentsTest : public ::testing::Test
{
protected:
   using EllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using EllpackSegmentsTypes =
   ::testing::Types< typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Host, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Host, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Host, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Host, long >::BaseType

#if defined( __CUDACC__ )
                     ,
                     typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Cuda, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Cuda, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Cuda, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Cuda, long >::BaseType
#elif defined( __HIP__ )
                     ,
                     typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Hip, int >::BaseType,
                     typename TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Hip, long >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Hip, int >::BaseType,
                     typename TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Hip, long >::BaseType
#endif
                     >;

TYPED_TEST_SUITE( EllpackReduceSegmentsTest, EllpackSegmentsTypes );

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegments_MaximumInSegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_reduceSegments_MaximumInSegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegments_MaximumInSegments_short_fetch )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_reduceSegments_MaximumInSegments_short_fetch< EllpackSegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegmentsWithArgument_MaximumInSegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_reduceSegmentsWithArgument_MaximumInSegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexes_MaximumInSegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_reduceSegmentsWithSegmentIndexes_MaximumInSegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegmentsIf_MaximumInSegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_reduceSegmentsIf_MaximumInSegments< EllpackSegmentsType >();
}

TYPED_TEST( EllpackReduceSegmentsTest, reduceSegmentsIfWithArg_MaximumInSegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_reduceSegmentsIfWithArgument_MaximumInSegments< EllpackSegmentsType >();
}

#include "../../../main.h"
