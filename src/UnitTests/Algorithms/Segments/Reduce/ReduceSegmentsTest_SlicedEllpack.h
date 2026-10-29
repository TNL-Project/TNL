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
using SlicedEllpackSegmentsTypes = ::testing::Types<
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

#include "ReduceSegmentsTestSuite.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P( SlicedEllpackSegments, ReduceSegmentsTest, SlicedEllpackSegmentsTypes );

// Reduction over a range with end < begin must not touch any segment.
template< typename Segments >
void
test_reduceSegments_ReversedRange()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 100;
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount, 3 );
   Segments segments( segmentsSizes );

   for( const auto& [ launch_config, tag ] : reductionLaunchConfigurations( segments ) ) {
      SCOPED_TRACE( tag );

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > result( segmentsCount, -1 );
      auto result_view = result.getView();
      auto fetch = [] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
      {
         return 1;
      };
      auto keep = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType res ) mutable
      {
         result_view[ segmentIdx ] = res;
      };
      TNL::Algorithms::Segments::reduceSegments(
         segments, (IndexType) 64, (IndexType) 10, fetch, TNL::Plus{}, keep, 0, launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ )
         EXPECT_EQ( result.getElement( i ), -1 ) << "segmentIdx = " << i;
   }
}

template< typename Segments >
class SlicedEllpackReduceSegmentsTest : public ::testing::Test
{};

TYPED_TEST_SUITE( SlicedEllpackReduceSegmentsTest, SlicedEllpackSegmentsTypes );

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegments_ReversedRange )
{
   test_reduceSegments_ReversedRange< TypeParam >();
}

#include "../../../main.h"
