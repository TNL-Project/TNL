#include <TNL/Algorithms/Segments/SlicedEllpack.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SlicedEllpackSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( SlicedEllpackSegmentsTest, SlicedEllpackSegmentsTypes );

TYPED_TEST( SlicedEllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_SetSegmentsSizes_EqualSizes< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackSegmentsTest, findInSegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_findInSegments< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackSegmentsTest, findInSegmentsWithIndexes )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_findInSegmentsWithIndexes< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackSegmentsTest, findInSegmentsIf )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_findInSegmentsIf< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackSegmentsTest, sortSegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_sortSegments< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackSegmentsTest, sortSegmentsWithSegmentIndexes )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_sortSegmentsWithSegmentIndexes< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackSegmentsTest, sortSegmentsIf )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_sortSegmentsIf< SlicedEllpackSegmentsType >();
}

#include "../../main.h"
