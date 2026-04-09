#include <TNL/Algorithms/Segments/Ellpack.h>

#include <gtest/gtest.h>

// types for which MatrixTest is instantiated
using EllpackSegmentsTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Host, int >,
   TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Host, long >,
   TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Host, int >,
   TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Host, long >
#elif defined( __CUDACC__ )
   TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Cuda, int >,
   TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Cuda, long >,
   TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Cuda, int >,
   TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
   TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Hip, int >,
   TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Hip, long >,
   TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Hip, int >,
   TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Hip, long >
#endif
   >;

#include "ReduceSegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( EllpackSegments, ReduceSegmentsTest, EllpackSegmentsTypes );

#include "../../../main.h"
