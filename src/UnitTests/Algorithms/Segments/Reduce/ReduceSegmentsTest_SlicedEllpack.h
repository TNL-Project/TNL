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

// Segments of whole slices are empty, i.e. the slices have zero segment size.
// All segments of such slices must be reported as empty by the reductions with argument.
template< typename Segments >
void
test_reduceSegmentsWithArgument_EmptySlices()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   // Blocks of 64 segments alternate between non-empty and empty segments,
   // so each empty block covers whole slices for all tested slice sizes.
   const IndexType segmentsCount = 256;
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType& value )
      {
         value = ( idx / 64 ) % 2 == 0 ? idx % 5 + 1 : 0;
      } );
   Segments segments( segmentsSizes );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentIndexes( segmentsCount );
   segmentIndexes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType& value )
      {
         value = idx;
      } );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > emptyFlags( segmentsCount, -1 );
   auto emptyFlags_view = emptyFlags.getView();
   auto fetch = [] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
   {
      return 1;
   };

   auto check = [ & ]()
   {
      for( IndexType i = 0; i < segmentsCount; i++ )
         EXPECT_EQ( emptyFlags.getElement( i ), segmentsSizes.getElement( i ) == 0 ) << "segmentIdx = " << i;
   };

   auto keep = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType res, bool emptySegment ) mutable
   {
      emptyFlags_view[ segmentIdx ] = emptySegment;
   };
   TNL::Algorithms::Segments::reduceAllSegmentsWithArgument( segments, fetch, TNL::MaxWithArg{}, keep );
   check();

   emptyFlags_view = -1;
   auto keepWithIndexes =
      [ = ] __cuda_callable__(
         IndexType indexOfSegmentIdx, IndexType segmentIdx, IndexType localIdx, IndexType res, bool emptySegment ) mutable
   {
      emptyFlags_view[ segmentIdx ] = emptySegment;
   };
   TNL::Algorithms::Segments::reduceSegmentsWithArgument( segments, segmentIndexes, fetch, TNL::MaxWithArg{}, keepWithIndexes );
   check();
}

template< typename Segments >
class SlicedEllpackReduceSegmentsTest : public ::testing::Test
{};

TYPED_TEST_SUITE( SlicedEllpackReduceSegmentsTest, SlicedEllpackSegmentsTypes );

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegments_ReversedRange )
{
   test_reduceSegments_ReversedRange< TypeParam >();
}

TYPED_TEST( SlicedEllpackReduceSegmentsTest, reduceSegmentsWithArgument_EmptySlices )
{
   test_reduceSegmentsWithArgument_EmptySlices< TypeParam >();
}

#include "../../../main.h"
