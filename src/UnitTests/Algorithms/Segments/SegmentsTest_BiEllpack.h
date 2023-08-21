#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/SegmentsReductionKernels/BiEllpackKernel.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class BiEllpackSegmentsTest : public ::testing::Test
{
protected:
   using BiEllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using BiEllpackSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Host, int >,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Host, long >
#ifdef __CUDACC__
                                                 ,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Cuda, int >,
                                                 TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Cuda, long >
#endif
                                                 >;

TYPED_TEST_SUITE( BiEllpackSegmentsTest, BiEllpackSegmentsTypes );

TYPED_TEST( BiEllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;

   test_SetSegmentsSizes_EqualSizes< BiEllpackSegmentsType >();
}

TYPED_TEST( BiEllpackSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using BiEllpackSegmentsType = typename TestFixture::BiEllpackSegmentsType;
   using Kernel = TNL::Algorithms::SegmentsReductionKernels::BiEllpackKernel< typename BiEllpackSegmentsType::IndexType,
                                                                              typename BiEllpackSegmentsType::DeviceType >;

   test_reduceAllSegments_MaximumInSegments< BiEllpackSegmentsType, Kernel >();
}

#include "../../main.h"
