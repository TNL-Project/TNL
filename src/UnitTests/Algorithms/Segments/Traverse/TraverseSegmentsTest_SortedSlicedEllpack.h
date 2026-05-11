#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>

#include <gtest/gtest.h>

// types for which MatrixTest is instantiated
using SortedSlicedEllpackSegmentsTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, long > >
#elif defined( __CUDACC__ )
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Cuda, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Cuda, long > >
#elif defined( __HIP__ )
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Hip, long > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Hip, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Hip, long > >
#endif
   >;

#include "TraverseSegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( SortedSlicedEllpackSegments, TraverseSegmentsTest, SortedSlicedEllpackSegmentsTypes );

#include "../../../main.h"
