#include <TNL/Algorithms/Segments/ChunkedEllpack.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class ChunkedEllpackSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( ChunkedEllpackSegmentsTest, ChunkedEllpackSegmentsTypes );

TYPED_TEST( ChunkedEllpackSegmentsTest, isSegments )
{
   test_isSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   test_setSegmentsSizes_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, findInSegments )
{
   test_findInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, findInSegmentsWithIndexes )
{
   test_findInSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, findInSegmentsIf )
{
   test_findInSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, sortSegments )
{
   test_sortSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, sortSegmentsWithSegmentIndexes )
{
   test_sortSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, sortSegmentsIf )
{
   test_sortSegmentsIf< typename TestFixture::SegmentsType >();
}

#include "../../main.h"
