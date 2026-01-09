#include <TNL/Algorithms/Segments/ChunkedEllpack.h>

#include <gtest/gtest.h>

// types for which MatrixTest is instantiated
using ChunkedEllpackSegmentsTypes =
   ::testing::Types< TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Host, int >,
                     TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Host, long >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Host, int >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                     ,
                     TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Cuda, int >,
                     TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Cuda, long >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Cuda, int >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                     ,
                     TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Hip, int >,
                     TNL::Algorithms::Segments::RowMajorChunkedEllpack< TNL::Devices::Hip, long >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Hip, int >,
                     TNL::Algorithms::Segments::ColumnMajorChunkedEllpack< TNL::Devices::Hip, long >
#endif
                     >;

#include "ReduceSegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( ChunkedEllpackSegments, ReduceSegmentsTest, ChunkedEllpackSegmentsTypes );

#include "../../../main.h"
