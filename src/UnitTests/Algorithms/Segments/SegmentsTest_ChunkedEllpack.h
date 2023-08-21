#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/SegmentsReductionKernels/ChunkedEllpackKernel.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class ChunkedEllpackSegmentsTest : public ::testing::Test
{
protected:
   using ChunkedEllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using ChunkedEllpackSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Host, int >,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Host, long >
#ifdef __CUDACC__
                                                      ,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Cuda, int >,
                                                      TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Cuda, long >
#endif
                                                      >;

TYPED_TEST_SUITE( ChunkedEllpackSegmentsTest, ChunkedEllpackSegmentsTypes );

TYPED_TEST( ChunkedEllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;

   test_SetSegmentsSizes_EqualSizes< ChunkedEllpackSegmentsType >();
}

TYPED_TEST( ChunkedEllpackSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using ChunkedEllpackSegmentsType = typename TestFixture::ChunkedEllpackSegmentsType;
   using Kernel =
      TNL::Algorithms::SegmentsReductionKernels::ChunkedEllpackKernel< typename ChunkedEllpackSegmentsType::IndexType,
                                                                       typename ChunkedEllpackSegmentsType::DeviceType >;

   test_reduceAllSegments_MaximumInSegments< ChunkedEllpackSegmentsType, Kernel >();
}

#include "../../main.h"
