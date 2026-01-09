#include <TNL/Algorithms/Segments/AdaptiveCSR.h>

#include <gtest/gtest.h>

// types for which MatrixTest is instantiated
using SortedAdaptiveCSRSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Host, int >,
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

#include "TraverseSegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( SortedAdaptiveCSRSegments, TraverseSegmentsTest, SortedAdaptiveCSRSegmentsTypes );

#include "../../../main.h"
