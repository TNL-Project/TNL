#include <TNL/Algorithms/Segments/Ellpack.h>

#include "TraverseSegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class EllpackTraverseSegmentsTest : public ::testing::Test
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

TYPED_TEST_SUITE( EllpackTraverseSegmentsTest, EllpackSegmentsTypes );

TYPED_TEST( EllpackTraverseSegmentsTest, forElements_EmptySegments )
{
   test_forElements_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElements_EqualSizes )
{
   test_forElements_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElements )
{
   test_forElements< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElementsIf )
{
   test_forElementsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElementsWithSegmentIndexes_EmptySegments )
{
   test_forElementsWithSegmentIndexes_EmptySegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forElementsWithSegmentIndexes )
{
   test_forElementsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forSegments )
{
   test_forSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forSegmentsWithIndexes )
{
   test_forSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forSegmentsIf )
{
   test_forSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( EllpackTraverseSegmentsTest, forSegmentsSequential )
{
   test_forSegmentsSequential< typename TestFixture::SegmentsType >();
}

#include "../../../main.h"
