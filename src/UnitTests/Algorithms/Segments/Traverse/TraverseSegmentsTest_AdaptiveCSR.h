#include <TNL/Algorithms/Segments/AdaptiveCSR.h>

#include <gtest/gtest.h>

// types for which MatrixTest is instantiated
using AdaptiveCSRSegmentsTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Host, int >,
   TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Host, long >
#elif defined( __CUDACC__ )
   TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Cuda, int >,
   TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
   TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Hip, int >,
   TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Hip, long >
#endif
   >;

#include "TraverseSegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( AdaptiveCSRSegments, TraverseSegmentsTest, AdaptiveCSRSegmentsTypes );

#include "../../../main.h"
