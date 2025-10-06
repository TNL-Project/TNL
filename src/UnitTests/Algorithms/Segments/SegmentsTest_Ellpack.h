#include <TNL/Algorithms/Segments/Ellpack.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class EllpackSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( EllpackSegmentsTest, EllpackSegmentsTypes );

TYPED_TEST( EllpackSegmentsTest, isSegments )
{
   test_isSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, getView )
{
   test_getView< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   test_setSegmentsSizes_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, setSegmentsSizes_EqualSizes_EllpackOnly )
{
   test_setSegmentsSizes_EqualSizes_EllpackOnly< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, findInSegments )
{
   test_findInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, findInSegmentsWithIndexes )
{
   test_findInSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, findInSegmentsIf )
{
   test_findInSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, sortSegments )
{
   test_sortSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, sortSegmentsWithSegmentIndexes )
{
   test_sortSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, sortSegmentsIf )
{
   test_sortSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, scanSegments )
{
   test_scanSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, scanSegmentsWithIndexes )
{
   test_scanSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, scanSegmentsIf )
{
   test_scanSegmentsIf< typename TestFixture::SegmentsType >();
}

#include "../../main.h"
