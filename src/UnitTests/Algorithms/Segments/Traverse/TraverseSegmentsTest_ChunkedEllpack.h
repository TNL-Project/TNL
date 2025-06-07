#include <TNL/Algorithms/Segments/ChunkedEllpack.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class ChunkedEllpackTraverseSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using ChunkedEllpackSegmentsTypes =
   ::testing::Types< TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Host, int >,
                     TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Host, long >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Host, int >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                     ,
                     TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Cuda, int >,
                     TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Cuda, long >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Cuda, int >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                     ,
                     TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Hip, int >,
                     TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Hip, long >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Hip, int >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Hip, long >
#endif
                     >;

TYPED_TEST_SUITE( ChunkedEllpackTraverseSegmentsTest, ChunkedEllpackSegmentsTypes );

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElements_EmptySegments )
{
   test_forElements_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElements_EqualSizes )
{
   test_forElements_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElements )
{
   test_forElements< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElementsIf )
{
   test_forElementsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   test_forElementsWithSegmentIndexes_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   test_forElementsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forSegments )
{
   test_forSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forSegmentsWithIndexes )
{
   test_forSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forSegmentsIf )
{
   test_forSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackTraverseSegmentsTest, forSegmentsSequential )
{
   test_forSegmentsSequential< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
