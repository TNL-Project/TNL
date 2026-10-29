// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Kernel for the Uniform thread mapping, traversing a
 * contiguous range of segments.
 *
 * The same number of threads (`threadsPerSegment`) is assigned to every
 * segment in the range [`begin`, `end`).
 * All segments of a slice have the same size, which is read from the slice
 * segment sizes, so no per-segment offsets are needed.
 *
 * - For the row-major organization, the elements of a segment are contiguous in
 *   memory, so the threads of a segment process them with stride
 *   `threadsPerSegment` and neighbouring threads access neighbouring elements.
 * - For the column-major organization, the elements with the same local index
 *   of all segments of a slice are contiguous, so the
 *   `SliceSize * threadsPerSegment` threads assigned to a slice are mapped such
 *   that neighbouring threads process neighbouring segments and the memory
 *   accesses are coalesced across the segments.
 *   The grid starts at the first slice of the range, so the threads of the
 *   segments outside [`begin`, `end`) do nothing.
 *
 * See \ref forElements_CSR_Uniform_WithIndexes for the paper that inspired the
 * uniform number of threads per segment.
 */
template< typename SegmentsConstView, typename Index, typename Function, ElementsOrganization Organization, int SliceSize >
__global__
void
forElements_SlicedEllpack_Uniform(
   const Index gridIdx,
   const Index threadsPerSegment,
   const SegmentsConstView segments,
   Index begin,
   Index end,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   if constexpr( Organization == RowMajorOrder ) {
      const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
      if( segmentIdx >= end )
         return;

      const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %

      const Index sliceIdx = segmentIdx / SliceSize;
      const Index segmentInSliceIdx = segmentIdx % SliceSize;
      const Index segmentSize = segments.getSliceSegmentSizesView()[ sliceIdx ];

      const Index beginIdx = segments.getSliceOffsetsView()[ sliceIdx ] + segmentInSliceIdx * segmentSize;
      const Index endIdx = beginIdx + segmentSize;
      TNL_ASSERT_EQ( beginIdx, segments.getGlobalIndex( segmentIdx, 0 ), "" );

      Index localIdx = laneIdx;
      for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
         TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( segmentIdx, localIdx ), "" );
         TNL_ASSERT_LT( globalIdx, endIdx, "" );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( callableArgumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
         localIdx += threadsPerSegment;
      }
   }
   else {  // ColumnMajorOrder
      const Index firstSliceIdx = begin / SliceSize;
      const Index sliceIdx = firstSliceIdx + Backend::getGlobalThreadIdx_x( gridIdx ) / ( SliceSize * threadsPerSegment );
      const Index inSliceIdx = Backend::getGlobalThreadIdx_x( gridIdx ) % ( SliceSize * threadsPerSegment );
      const Index inSliceSegmentIdx = inSliceIdx % SliceSize;
      const Index segmentIdx = sliceIdx * SliceSize + inSliceSegmentIdx;

      if( segmentIdx < begin || segmentIdx >= end )
         return;

      TNL_ASSERT_LT( sliceIdx, segments.getSliceSegmentSizesView().getSize(), "" );
      TNL_ASSERT_LT( inSliceSegmentIdx, SliceSize, "" );
      TNL_ASSERT_LT( segmentIdx, segments.getSegmentCount(), "" );
      Index localIdx = inSliceIdx / SliceSize;

      const Index beginIdx = segments.getSliceOffsetsView()[ sliceIdx ] + inSliceSegmentIdx + localIdx * SliceSize;
      const Index endIdx = segments.getSliceOffsetsView()[ sliceIdx + 1 ];
      TNL_ASSERT_LE( endIdx, segments.getStorageSize(), "" );  // equality is for the case when both values equal 0

      const Index step = threadsPerSegment * SliceSize;
      for( Index globalIdx = beginIdx; globalIdx < endIdx; globalIdx += step ) {
         TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( segmentIdx, localIdx ), "" );
         TNL_ASSERT_LT( globalIdx, endIdx, "" );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( callableArgumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
         localIdx += threadsPerSegment;
      }
   }
#endif
}

/**
 * \brief Kernel for the Uniform thread mapping, traversing the segments
 * given explicitly by `segmentIndexes`.
 *
 * `threadsPerSegment` consecutive threads are assigned to
 * every entry of `segmentIndexes`.
 * Since the segments are arbitrary, the threads are always grouped by segment,
 * also for the column-major organization, where the threads of a segment access
 * its elements with stride `threadsPerSegment * SliceSize`.
 * See \ref forElements_SlicedEllpack_Uniform for the details of the format.
 */
template<
   typename SegmentsConstView,
   typename ArrayView,
   typename Index,
   typename Function,
   ElementsOrganization Organization,
   int SliceSize >
__global__
void
forElements_SlicedEllpack_Uniform_WithIndexes(
   const Index gridIdx,
   const Index threadsPerSegment,
   const SegmentsConstView segments,
   const ArrayView segmentIndexes,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index idx = Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
   if( idx >= segmentIndexes.getSize() )
      return;
   TNL_ASSERT_GE( idx, 0, "" );
   TNL_ASSERT_LT( idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ idx ];
   TNL_ASSERT_GE( segmentIdx, 0, "Wrong index segment index - smaller that 0." );
   TNL_ASSERT_LT( segmentIdx, segments.getSegmentCount(), "Wrong index segment index - larger that the number of indexes." );
   const Index sliceIdx = segmentIdx / SliceSize;
   const Index inSliceOffset = segmentIdx % SliceSize;

   const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %
   const Index segmentSize = segments.getSliceSegmentSizesView()[ sliceIdx ];

   if constexpr( Organization == RowMajorOrder ) {
      Index globalIdx = segments.getSliceOffsetsView()[ sliceIdx ] + inSliceOffset * segmentSize + laneIdx;
      for( Index localIdx = laneIdx; localIdx < segmentSize; localIdx += threadsPerSegment ) {
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( callableArgumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
         globalIdx += threadsPerSegment;
      }
   }
   else {
      Index globalIdx = segments.getSliceOffsetsView()[ sliceIdx ] + inSliceOffset + laneIdx * SliceSize;
      for( Index localIdx = laneIdx; localIdx < segmentSize; localIdx += threadsPerSegment ) {
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( callableArgumentCount< Function >() == 3 )
            function( segmentIdx, localIdx, globalIdx );
         else
            function( segmentIdx, globalIdx );
         globalIdx += threadsPerSegment * SliceSize;
      }
   }
#endif
}

/**
 * \brief Kernel for the Uniform thread mapping, traversing a contiguous
 * range of segments that satisfy `condition`.
 *
 * `threadsPerSegment` consecutive threads are assigned to every
 * segment of the range [`begin`, `end`) and they skip the segment
 * if `condition` is not satisfied.
 * Like in \ref forElements_SlicedEllpack_Uniform_WithIndexes, the threads are
 * grouped by segment also for the column-major organization.
 * See \ref forElements_SlicedEllpack_Uniform for the details of the format.
 */
template<
   typename SegmentsConstView,
   typename Index,
   typename Condition,
   typename Function,
   ElementsOrganization Organization,
   int SliceSize,
   int BlockSize = 256 >
__global__
void
forElements_SlicedEllpack_Uniform_If(
   const Index gridIdx,
   const Index threadsPerSegment,
   const SegmentsConstView segments,
   Index begin,
   Index end,
   Condition condition,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
   if( segmentIdx >= end || ! condition( segmentIdx ) )
      return;

   const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %
   const Index segmentSize = segments.getSegmentSize( segmentIdx );

   if constexpr( Organization == RowMajorOrder ) {
      Index globalIdx = segments.getGlobalIndex( segmentIdx, 0 );
      const Index endIdx = globalIdx + segmentSize;
      globalIdx += laneIdx;
      if constexpr( callableArgumentCount< Function >() == 3 ) {
         Index localIdx = laneIdx;
         for( ; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, localIdx, globalIdx );
            localIdx += threadsPerSegment;
         }
      }
      else {  // callableArgumentCount< Function >() == 2
         for( ; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, globalIdx );
         }
      }
   }
   else {  // Organization == ColumnMajorOrder
      Index globalIdx = segments.getGlobalIndex( segmentIdx, laneIdx );
      const Index endIdx = segments.getGlobalIndex( segmentIdx, segmentSize );
      if constexpr( callableArgumentCount< Function >() == 3 ) {
         Index localIdx = laneIdx;
         for( ; globalIdx < endIdx; globalIdx += SliceSize * threadsPerSegment ) {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, localIdx, globalIdx );
            localIdx += threadsPerSegment;
         }
      }
      else {
         for( ; globalIdx < endIdx; globalIdx += SliceSize * threadsPerSegment ) {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            function( segmentIdx, globalIdx );
         }
      }
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
