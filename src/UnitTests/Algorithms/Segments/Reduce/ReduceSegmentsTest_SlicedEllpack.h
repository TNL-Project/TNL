#include <TNL/Algorithms/Segments/SlicedEllpack.h>

#include <gtest/gtest.h>

template< typename Device, typename Index, int SliceSize >
using RowMajorSlicedEllpackType = TNL::Algorithms::Segments::
   RowMajorSlicedEllpack< Device, Index, typename TNL::Allocators::Default< Device >::template Allocator< Index >, SliceSize >;

template< typename Device, typename Index, int SliceSize >
using ColumnMajorSlicedEllpackType = TNL::Algorithms::Segments::ColumnMajorSlicedEllpack<
   Device,
   Index,
   typename TNL::Allocators::Default< Device >::template Allocator< Index >,
   SliceSize >;

// types for which MatrixTest is instantiated
using SlicedEllpackSegmentsTypes =
   ::testing::Types< TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, int >,
                     TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, long >,
                     TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, int >,
                     TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                     ,
                     RowMajorSlicedEllpackType< TNL::Devices::Cuda, int, 32 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Cuda, int, 8 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Cuda, int, 1 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Cuda, long int, 32 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Cuda, long int, 1 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Cuda, int, 32 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Cuda, int, 8 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Cuda, int, 1 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Cuda, long int, 32 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Cuda, long int, 1 >
#elif defined( __HIP__ )
                     ,
                     RowMajorSlicedEllpackType< TNL::Devices::Hip, int, 32 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Hip, int, 8 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Hip, int, 1 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Hip, long int, 32 >,
                     RowMajorSlicedEllpackType< TNL::Devices::Hip, long int, 1 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Hip, int, 32 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Hip, int, 8 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Hip, int, 1 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Hip, long int, 32 >,
                     ColumnMajorSlicedEllpackType< TNL::Devices::Hip, long int, 1 >
#endif
                     >;

#include "ReduceSegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( SlicedEllpackSegments, ReduceSegmentsTest, SlicedEllpackSegmentsTypes );

#include "../../../main.h"
