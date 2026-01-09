#pragma once

#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <gtest/gtest.h>

// Types for which SegmentsTest is instantiated - SlicedEllpack segments
using SlicedEllpackSegmentsTypes =
   ::testing::Types< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, int >,
                     TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, long >,
                     TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, int >,
                     TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                     ,
                     TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, int >,
                     TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, long >,
                     TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Cuda, int >,
                     TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                     ,
                     TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, int >,
                     TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, long >,
                     TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Hip, int >,
                     TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Hip, long >
#endif
                     >;

#include "SegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( SlicedEllpackSegments, SegmentsTest, SlicedEllpackSegmentsTypes );

#include "../../main.h"
