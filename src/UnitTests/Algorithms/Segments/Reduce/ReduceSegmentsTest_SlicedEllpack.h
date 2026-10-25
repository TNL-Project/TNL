#pragma once

// Types of the SlicedEllpack segments for the ReduceSegmentsTest, included in the generated
// ReduceSegmentsTest_SlicedEllpack sources (see tnl_add_format_tests in CMakeLists.txt)

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

// more slice sizes are tested on GPUs
using SlicedEllpackReduceSegmentsTestTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, int >,
   TNL::Algorithms::Segments::RowMajorSlicedEllpack< TNL::Devices::Host, long >,
   TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, int >,
   TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< TNL::Devices::Host, long >
#elif defined( __CUDACC__ )
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
