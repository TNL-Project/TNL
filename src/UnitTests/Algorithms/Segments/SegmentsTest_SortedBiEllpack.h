#pragma once

#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>
#include <TNL/Algorithms/Segments/SortedSegmentsView.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SortedBiEllpackSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SortedBiEllpackSegmentsTypes = ::testing::Types<
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Host, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Host, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Host, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Host, long > >
#if defined( __CUDACC__ )
   ,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Cuda, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Cuda, long > >
#elif defined( __HIP__ )
   ,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Hip, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Hip, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Hip, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Hip, long > >
#endif
   >;

TYPED_TEST_SUITE( SortedBiEllpackSegmentsTest, SortedBiEllpackSegmentsTypes );

TYPED_TEST( SortedBiEllpackSegmentsTest, isSegments )
{
   test_isSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   test_SetSegmentsSizes_EqualSizes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackSegmentsTest, findInSegments )
{
   test_findInSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackSegmentsTest, findInSegmentsWithIndexes )
{
   test_findInSegmentsWithIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackSegmentsTest, findInSegmentsIf )
{
   test_findInSegmentsIf< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackSegmentsTest, sortSegments )
{
   test_sortSegments< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackSegmentsTest, sortSegmentsWithSegmentIndexes )
{
   test_sortSegmentsWithSegmentIndexes< typename TestFixture::SegmentsType >();
}

TYPED_TEST( SortedBiEllpackSegmentsTest, sortSegmentsIf )
{
   test_sortSegmentsIf< typename TestFixture::SegmentsType >();
}

#include "../../main.h"
