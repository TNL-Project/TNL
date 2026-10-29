// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/detail/CudaScanKernel.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Kernel for the PerElement thread mapping, traversing
 * a contiguous range of segments.
 *
 * Every block processes `SegmentsPerBlock` consecutive segments of the
 * range [`begin`, `end`) and the elements of these segments are divided
 * evenly among the threads of the block.
 * Since all Ellpack segments have the same size, the segment index and the
 * local index of an element are obtained by a division and a modulo.
 * For the row-major organization, the consecutive threads process the
 * consecutive elements of one segment; for the column-major organization, they
 * process the elements with the same local index in consecutive segments, so
 * that the memory accesses are coalesced in both cases.
 *
 * See \ref forElements_CSR_PerElement for the general
 * idea of the PerElement mapping.
 */
template<
   typename SegmentsView,
   typename Index,
   typename Function,
   ElementsOrganization Organization,
   int SegmentsPerBlock = 256,
   int BlockSize = 256 >
__global__
void
forElements_Ellpack_PerElement( Index gridIdx, SegmentsView segments, Index begin, Index end, Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentSize = segments.getSegmentSize();
   const Index firstSegmentIdx = begin + Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock;
   const Index segmentsInBlock = min( end - firstSegmentIdx, (Index) SegmentsPerBlock );
   if( segmentSize == 0 || segmentsInBlock <= 0 )
      return;

   const Index elementsInBlock = segmentsInBlock * segmentSize;
   for( Index idx = threadIdx.x; idx < elementsInBlock; idx += BlockSize ) {
      Index segmentIdx;
      Index localIdx;
      if constexpr( Organization == Algorithms::Segments::RowMajorOrder ) {
         segmentIdx = firstSegmentIdx + idx / segmentSize;
         localIdx = idx % segmentSize;
      }
      else {  // ColumnMajorOrder
         segmentIdx = firstSegmentIdx + idx % segmentsInBlock;
         localIdx = idx / segmentsInBlock;
      }
      TNL_ASSERT_LT( segmentIdx, end, "" );
      TNL_ASSERT_LT( localIdx, segmentSize, "" );
      const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
      if constexpr( callableArgumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
   }
#endif
}

/**
 * \brief Kernel for the PerElement thread mapping, traversing the segments
 * given explicitly by `segmentIndexes`.
 *
 * It works the same way as \ref forElements_Ellpack_PerElement, only every
 * block processes `SegmentsPerBlock` consecutive entries of `segmentIndexes`.
 */
template<
   typename SegmentsConstView,
   typename ArrayView,
   typename Index,
   typename Function,
   ElementsOrganization Organization,
   int SegmentsPerBlock = 256,
   int BlockSize = 256 >
__global__
void
forElements_Ellpack_PerElement_WithIndexes(
   Index gridIdx,
   SegmentsConstView segments,
   ArrayView segmentIndexes,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentSize = segments.getSegmentSize();
   const Index firstEntry = Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock;
   const Index segmentsInBlock = min( segmentIndexes.getSize() - firstEntry, (Index) SegmentsPerBlock );
   if( segmentSize == 0 || segmentsInBlock <= 0 )
      return;

   const Index elementsInBlock = segmentsInBlock * segmentSize;
   for( Index idx = threadIdx.x; idx < elementsInBlock; idx += BlockSize ) {
      Index entry;
      Index localIdx;
      if constexpr( Organization == Algorithms::Segments::RowMajorOrder ) {
         entry = firstEntry + idx / segmentSize;
         localIdx = idx % segmentSize;
      }
      else {  // ColumnMajorOrder
         entry = firstEntry + idx % segmentsInBlock;
         localIdx = idx / segmentsInBlock;
      }
      TNL_ASSERT_LT( entry, segmentIndexes.getSize(), "" );
      TNL_ASSERT_LT( localIdx, segmentSize, "" );
      const Index segmentIdx = segmentIndexes[ entry ];
      const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
      if constexpr( callableArgumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
   }
#endif
}

/**
 * \brief Kernel for the PerElement thread mapping, traversing the segments of a
 * contiguous range that satisfy `condition`.
 *
 * Every block evaluates `condition` for `SegmentsPerBlock` consecutive
 * segments of the range [`begin`, `end`) and compacts the indexes of
 * the segments satisfying it into shared memory using a block-wide
 * inclusive scan of the condition values.
 * The elements of the active segments are then divided evenly among the threads
 * of the block; since all segments have the same size, the segment and the
 * local index of each element are obtained by a division and a modulo.
 * The template parameter `SliceSize` is not used.
 *
 * See \ref forElements_CSR_PerElement for the general
 * idea of the PerElement mapping.
 */
template<
   typename SegmentsView,
   typename Index,
   typename Condition,
   typename Function,
   ElementsOrganization Organization,
   int SliceSize,
   int SegmentsPerBlock,
   int BlockSize = 256 >
__global__
void
forElements_Ellpack_PerElement_If(
   Index gridIdx,
   SegmentsView segments,
   const Index begin,
   const Index end,
   Condition condition,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using InclusiveCudaScan =
      Algorithms::detail::CudaBlockScanShfl< Algorithms::detail::ScanType::Inclusive, BlockSize, Plus, Index >;
   using InclusiveScanStorage = typename InclusiveCudaScan::Storage;  // TODO: Storage should not depend on the scan type

   __shared__ InclusiveScanStorage inclusive_scan_storage;
   __shared__ Index conditions[ SegmentsPerBlock ];
   __shared__ Index shared_segment_indexes[ SegmentsPerBlock ];
   __shared__ Index segmentSize;

   if( threadIdx.x == 0 )
      segmentSize = segments.getSegmentSize();
   const Index segmentIdx = begin + Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock + threadIdx.x;
   Index conditionValue = 0;
   if( segmentIdx < end && threadIdx.x < SegmentsPerBlock )
      conditionValue = condition( segmentIdx );

   // initialize all shared slots that any later phase may read: threads whose segment index
   // is out of range do not write anything below and their slots must still have a defined value
   if( threadIdx.x < SegmentsPerBlock ) {
      conditions[ threadIdx.x ] = 0;
      shared_segment_indexes[ threadIdx.x ] = 0;
   }
   __syncthreads();

   const Index v1 = InclusiveCudaScan::scan( Plus{}, (Index) 0, conditionValue, threadIdx.x, inclusive_scan_storage );
   if( threadIdx.x < SegmentsPerBlock && threadIdx.x < BlockSize )
      conditions[ threadIdx.x ] = v1;
   __syncthreads();

   __shared__ Index activeSegmentsCount;
   if( threadIdx.x == 0 )
      activeSegmentsCount = conditions[ SegmentsPerBlock - 1 ];
   const Index prevConditions = ( threadIdx.x == 0 || threadIdx.x > SegmentsPerBlock ) ? 0 : conditions[ threadIdx.x - 1 ];
   if( threadIdx.x < SegmentsPerBlock && conditions[ threadIdx.x ] != prevConditions )
      shared_segment_indexes[ conditions[ threadIdx.x ] - 1 ] = segmentIdx;
   __syncthreads();

   const Index last_idx = activeSegmentsCount * segments.getSegmentSize();
   if( activeSegmentsCount > 0 )
      TNL_ASSERT_LE( last_idx, segments.getStorageSize() - shared_segment_indexes[ 0 ], "" );

   Index idx = threadIdx.x;
   while( idx < last_idx ) {
      if constexpr( Organization == RowMajorOrder ) {
         const Index local_segmentIdx = idx / segmentSize;
         TNL_ASSERT_GE( local_segmentIdx, 0, "" );
         TNL_ASSERT_LT( local_segmentIdx, activeSegmentsCount, "" );
         TNL_ASSERT_LT( local_segmentIdx, SegmentsPerBlock, "" );

         const Index localIdx = idx % segmentSize;
         const Index segmentIdx = shared_segment_indexes[ local_segmentIdx ];
         const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
         TNL_ASSERT_GE( globalIdx, 0, "" );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( callableArgumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
      }
      else {  // ColumnMajorOrder
         const Index localIdx = idx / activeSegmentsCount;
         const Index segmentIdx = shared_segment_indexes[ idx % activeSegmentsCount ];
         const Index globalIdx = segments.getGlobalIndex( segmentIdx, localIdx );
         TNL_ASSERT_GE( globalIdx, 0, "" );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( callableArgumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
      }

      idx += BlockSize;
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
