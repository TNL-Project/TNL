#pragma once

#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include <gtest/gtest.h>

// Types for which SegmentsTest is instantiated - AdaptiveCSR segments
using AdaptiveCSRSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Host, int >,
                                                   TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                                   ,
                                                   TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Cuda, int >,
                                                   TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                                   ,
                                                   TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Hip, int >,
                                                   TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Hip, long >
#endif
                                                   >;

#include "SegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( AdaptiveCSRSegments, SegmentsTest, AdaptiveCSRSegmentsTypes );

#include "../../main.h"
