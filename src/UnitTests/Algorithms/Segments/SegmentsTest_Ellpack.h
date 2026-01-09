#pragma once

#include <TNL/Algorithms/Segments/Ellpack.h>
#include <gtest/gtest.h>

// Types for which SegmentsTest is instantiated - Ellpack segments
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

#include "SegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( EllpackSegments, SegmentsTest, EllpackSegmentsTypes );

// Ellpack-specific test fixture
template< typename Segments >
class EllpackSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

TYPED_TEST_SUITE( EllpackSegmentsTest, EllpackSegmentsTypes );

TYPED_TEST( EllpackSegmentsTest, setSegmentsSizes_EqualSizes_EllpackOnly )
{
   test_setSegmentsSizes_EqualSizes_EllpackOnly< typename TestFixture::SegmentsType >();
}

#include "../../main.h"
