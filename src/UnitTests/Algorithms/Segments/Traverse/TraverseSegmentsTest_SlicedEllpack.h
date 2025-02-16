#include <TNL/Algorithms/Segments/SlicedEllpack.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SlicedEllpackTraverseSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( SlicedEllpackTraverseSegmentsTest, SlicedEllpackSegmentsTypes );

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElements_EmptySegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_forElements_EmptySegments< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElements_EqualSizes )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_forElements_EqualSizes< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElements )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_forElements< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElementsIf )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_forElementsIf< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_forElementsWithSegmentIndexes_EmptySegments< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_forElementsWithSegmentIndexes< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackTraverseSegmentsTest, forSegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_forSegments< SlicedEllpackSegmentsType >();
}

#include "../../../main.h"
