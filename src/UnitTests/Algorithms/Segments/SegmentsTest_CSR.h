#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRAdaptiveKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRHybridKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRLightKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRScalarKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRVectorKernel.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class CSRSegmentsTest : public ::testing::Test
{
protected:
   using CSRSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using CSRSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::CSR< TNL::Devices::Host, int >,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Host, long >
#ifdef __CUDACC__
                                           ,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, int >,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, long >
#endif
                                           >;

TYPED_TEST_SUITE( CSRSegmentsTest, CSRSegmentsTypes );

TYPED_TEST( CSRSegmentsTest, setSegmentsSizes_EqualSizes )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

   test_SetSegmentsSizes_EqualSizes< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, reduceAllSegments_MaximumInSegments_CSRAdaptive )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   using Kernel = TNL::Algorithms::SegmentsReductionKernels::CSRAdaptiveKernel< typename CSRSegmentsType::IndexType,
                                                                                typename CSRSegmentsType::DeviceType >;
   test_reduceAllSegments_MaximumInSegments< CSRSegmentsType, Kernel >();
}

TYPED_TEST( CSRSegmentsTest, reduceAllSegments_MaximumInSegments_CSRHybrid )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   using Kernel = TNL::Algorithms::SegmentsReductionKernels::CSRHybridKernel< typename CSRSegmentsType::IndexType,
                                                                              typename CSRSegmentsType::DeviceType >;
   test_reduceAllSegments_MaximumInSegments< CSRSegmentsType, Kernel >();
}

TYPED_TEST( CSRSegmentsTest, reduceAllSegments_MaximumInSegments_CSRLight )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   using Kernel = TNL::Algorithms::SegmentsReductionKernels::CSRLightKernel< typename CSRSegmentsType::IndexType,
                                                                             typename CSRSegmentsType::DeviceType >;
   test_reduceAllSegments_MaximumInSegments< CSRSegmentsType, Kernel >();
}

TYPED_TEST( CSRSegmentsTest, reduceAllSegments_MaximumInSegments_CSRScalar )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   using Kernel = TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel< typename CSRSegmentsType::IndexType,
                                                                              typename CSRSegmentsType::DeviceType >;
   test_reduceAllSegments_MaximumInSegments< CSRSegmentsType, Kernel >();
}

TYPED_TEST( CSRSegmentsTest, reduceAllSegments_MaximumInSegments_CSRVector )
{
   using CSRSegmentsType = typename TestFixture::CSRSegmentsType;
   using Kernel = TNL::Algorithms::SegmentsReductionKernels::CSRVectorKernel< typename CSRSegmentsType::IndexType,
                                                                              typename CSRSegmentsType::DeviceType >;
   test_reduceAllSegments_MaximumInSegments< CSRSegmentsType, Kernel >();
}

#include "../../main.h"
