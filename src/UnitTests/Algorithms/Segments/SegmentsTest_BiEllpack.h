#include <TNL/Algorithms/Segments/BiEllpack.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class BiEllpackSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( BiEllpackSegmentsTest, BiEllpackSegmentsTypes );

TYPED_TEST( BiEllpackSegmentsTest, isSegments )
{
   test_isSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   test_setSegmentsSizes_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, findInSegments )
{
   test_findInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, findInSegmentsWithIndexes )
{
   test_findInSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, findInSegmentsIf )
{
   test_findInSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, sortSegments )
{
   test_sortSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, sortSegmentsWithSegmentIndexes )
{
   test_sortSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, sortSegmentsIf )
{
   test_sortSegmentsIf< typename TestFixture::SegmentsType >();
}

#include "../../main.h"
