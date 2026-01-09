#pragma once

#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <gtest/gtest.h>

// Types for which SegmentsTest is instantiated - BiEllpack segments
using BiEllpackSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Host, int >,
                                                 TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Host, long >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Host, int >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                                 ,
                                                 TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Cuda, int >,
                                                 TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Cuda, long >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Cuda, int >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                                 ,
                                                 TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Hip, int >,
                                                 TNL::Algorithms::Segments::RowMajorBiEllpack< TNL::Devices::Hip, long >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Hip, int >,
                                                 TNL::Algorithms::Segments::ColumnMajorBiEllpack< TNL::Devices::Hip, long >
#endif
                                                 >;

#include "SegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( BiEllpackSegments, SegmentsTest, BiEllpackSegmentsTypes );

#include "../../main.h"
