#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/SegmentsReductionKernels/SlicedEllpackKernel.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SlicedEllpackSegmentsTest : public ::testing::Test
{
protected:
   using SlicedEllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SlicedEllpackSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Host, int >,
                                                     TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Host, long >
#ifdef __CUDACC__
                                                     ,
                                                     TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Cuda, int >,
                                                     TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Cuda, long >
#endif
                                                     >;

TYPED_TEST_SUITE( SlicedEllpackSegmentsTest, SlicedEllpackSegmentsTypes );

TYPED_TEST( SlicedEllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

   test_SetSegmentsSizes_EqualSizes< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;
   using Kernel =
      TNL::Algorithms::SegmentsReductionKernels::SlicedEllpackKernel< typename SlicedEllpackSegmentsType::IndexType,
                                                                      typename SlicedEllpackSegmentsType::DeviceType >;

   test_reduceAllSegments_MaximumInSegments< SlicedEllpackSegmentsType, Kernel >();
}

#include "../../main.h"
