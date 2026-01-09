#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>

#include <gtest/gtest.h>

// types for which MatrixTest is instantiated
using SortedCSRSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Host, int >,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                                 ,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Cuda, int >,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                                 ,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Hip, int >,
                                                 TNL::Algorithms::Segments::SortedCSR< TNL::Devices::Hip, long >
#endif
                                                 >;

#include "TraverseSegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( SortedCSRSegments, TraverseSegmentsTest, SortedCSRSegmentsTypes );

#include "../../../main.h"
