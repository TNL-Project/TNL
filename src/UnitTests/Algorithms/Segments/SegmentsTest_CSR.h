#pragma once

#include <TNL/Algorithms/Segments/CSR.h>
#include <gtest/gtest.h>

// Types for which SegmentsTest is instantiated - CSR segments
using CSRSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::CSR< TNL::Devices::Host, int >,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                           ,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, int >,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                           ,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Hip, int >,
                                           TNL::Algorithms::Segments::CSR< TNL::Devices::Hip, long >
#endif
                                           >;

#include "SegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( CSRSegments, SegmentsTest, CSRSegmentsTypes );

#include "../../main.h"
