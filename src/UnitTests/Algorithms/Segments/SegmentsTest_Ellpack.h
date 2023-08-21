#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/SegmentsReductionKernels/EllpackKernel.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class EllpackSegmentsTest : public ::testing::Test
{
protected:
   using EllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using EllpackSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::Ellpack< TNL::Devices::Host, int >,
                                               TNL::Algorithms::Segments::Ellpack< TNL::Devices::Host, long >
#ifdef __CUDACC__
                                               ,
                                               TNL::Algorithms::Segments::Ellpack< TNL::Devices::Cuda, int >,
                                               TNL::Algorithms::Segments::Ellpack< TNL::Devices::Cuda, long >
#endif
                                               >;

TYPED_TEST_SUITE( EllpackSegmentsTest, EllpackSegmentsTypes );

TYPED_TEST( EllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_SetSegmentsSizes_EqualSizes< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, setSegmentsSizes_EqualSizes_EllpackOnly )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

   test_SetSegmentsSizes_EqualSizes_EllpackOnly< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, reduceAllSegments_MaximumInSegments )
{
   using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;
   using Kernel = TNL::Algorithms::SegmentsReductionKernels::EllpackKernel< typename EllpackSegmentsType::IndexType,
                                                                            typename EllpackSegmentsType::DeviceType >;

   test_reduceAllSegments_MaximumInSegments< EllpackSegmentsType, Kernel >();
}

#include "../../main.h"
