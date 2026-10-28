// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/detail/CudaScanKernel.h>
#include <TNL/Algorithms/find.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Kernel for the PerElement thread mapping, traversing a contiguous range of segments.
 *
 * Every block processes `SegmentsPerBlock / SliceSize` consecutive slices.
 * The offsets of the slices (relative to the first element of the block)
 * and the segment sizes in the slices are stored in shared memory.
 * All elements of the block's slices are divided evenly among the threads of the block
 * regardless of the segment boundaries;
 * each thread finds the slice its element belongs to by a binary search in the slice offsets
 * and computes the segment and the local index within the slice directly,
 * since all segments in a slice have the same size.
 * Elements of the segments outside of the range [`begin`, `end`)
 * (in the first and the last slice) are skipped.
 *
 * See \ref forElements_CSR_PerElement for the general idea of the PerElement mapping
 * and the literature reference.
 */
template<
   typename SegmentsConstView,
   typename Index,
   typename Function,
   ElementsOrganization Organization,
   int SliceSize,
   int SegmentsPerBlock = 256,
   int BlockSize = 256 >
__global__
void
forElements_SlicedEllpack_PerElement(
   const Index gridIdx,
   const SegmentsConstView segments,
   Index begin,
   Index end,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   static_assert( SegmentsPerBlock <= BlockSize, "Every segment of the block must have its own thread." );
   static_assert( SegmentsPerBlock % SliceSize == 0, "The block must process whole slices." );
   constexpr int SlicesPerBlock = SegmentsPerBlock / SliceSize;

   __shared__ Index shared_offsets[ SlicesPerBlock + 1 ];
   __shared__ Index shared_segments_size[ SlicesPerBlock ];

   __shared__ Index firstSliceIdx;              // the slice corresponding to the segment with index `begin`
   __shared__ Index lastSliceIdx;               // the slice corresponding to the segment with index `end - 1`
   __shared__ Index firstSegmentIdx;            // the index of the first segment in the first slice
   __shared__ Index firstSegmentInBlock;        // the index of the first segment being processed in this block of threads
   __shared__ Index firstSliceInBlockIdx;       // the index of the first slice being processed in this block of threads
   __shared__ Index firstSegmentInBlockOffset;  // the offset of the first segment in this block of threads
   __shared__ Index slicesInBlockCount;         // the number of slices being processed in this block of threads

   if( threadIdx.x == 0 ) {
      firstSliceIdx = begin / SliceSize;
      lastSliceIdx = ( end - 1 ) / SliceSize;
      firstSegmentIdx = firstSliceIdx * SliceSize;
   }
   __syncthreads();

   const Index segmentIdx = firstSegmentIdx + Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock + threadIdx.x;
   const Index segmentInSliceIdx = segmentIdx % SliceSize;
   const Index sliceIdx = segmentIdx / SliceSize;
   if( threadIdx.x == 0 ) {
      firstSegmentInBlock = segmentIdx;
      firstSliceInBlockIdx = firstSegmentInBlock / SliceSize;
      firstSegmentInBlockOffset = segments.getGlobalIndex( firstSegmentInBlock, 0 );  // segmentIdx
      slicesInBlockCount = min( lastSliceIdx - firstSliceInBlockIdx + 1, SlicesPerBlock );
   }
   __syncthreads();

   if( threadIdx.x < SegmentsPerBlock && segmentInSliceIdx == 0 && segmentIdx < end ) {
      shared_offsets[ sliceIdx - firstSliceInBlockIdx ] = segments.getGlobalIndex( segmentIdx, 0 ) - firstSegmentInBlockOffset;
      shared_segments_size[ sliceIdx - firstSliceInBlockIdx ] = segments.getSliceSegmentSizesView()[ sliceIdx ];
   }
   __syncthreads();
   if( threadIdx.x == 0 && segmentIdx < end ) {
      TNL_ASSERT_LT( slicesInBlockCount + firstSliceInBlockIdx - 1, segments.getSliceSegmentSizesView().getSize(), "" );
      shared_offsets[ slicesInBlockCount ] =
         shared_offsets[ slicesInBlockCount - 1 ]
         + SliceSize * segments.getSliceSegmentSizesView()[ slicesInBlockCount + firstSliceInBlockIdx - 1 ];
   }
   __syncthreads();

   const Index last_idx = shared_offsets[ slicesInBlockCount ];

   Index idx = threadIdx.x;
   while( idx < last_idx ) {
      auto [ found, local_sliceIdx ] = Algorithms::findUpperBound( shared_offsets, slicesInBlockCount + 1, idx );
      if( found ) {
         local_sliceIdx--;

         const Index inSliceIdx = idx - shared_offsets[ local_sliceIdx ];

         const Index local_segmentIdx = local_sliceIdx * SliceSize + inSliceIdx / shared_segments_size[ local_sliceIdx ];

         const Index globalIdx = firstSegmentInBlockOffset + idx;
         Index localIdx, currentSegmentIdx;
         if constexpr( Organization == Algorithms::Segments::RowMajorOrder ) {
            localIdx = inSliceIdx % shared_segments_size[ local_sliceIdx ];
            currentSegmentIdx = firstSegmentInBlock + local_segmentIdx;
         }
         else {  // ColumnMajorOrder
            localIdx = inSliceIdx / SliceSize;
            currentSegmentIdx = firstSegmentInBlock + local_sliceIdx * SliceSize + inSliceIdx % SliceSize;
         }

         if( currentSegmentIdx >= begin && currentSegmentIdx < end ) {
            TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( currentSegmentIdx, localIdx ), "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );

            if constexpr( callableArgumentCount< Function >() == 3 )
               function( currentSegmentIdx, localIdx, globalIdx );
            else
               function( currentSegmentIdx, globalIdx );
         }
      }
      idx += BlockSize;
   }

#endif
}

/**
 * \brief Kernel for the PerElement thread mapping, traversing the segments given explicitly by
 * \e segmentIndexes.
 *
 * Every block processes \e SegmentsPerBlock consecutive entries of \e segmentIndexes. The local
 * offsets of the segments within the block are computed by a block-wide exclusive scan of their
 * sizes, their global offsets are stored separately in shared memory. All elements of the block's
 * segments are then divided evenly among the threads of the block and each thread finds the segment
 * its element belongs to by a binary search in the local offsets.
 *
 * This is the same algorithm as \ref forElements_CSR_PerElement_WithIndexes, only the global index
 * of an element is computed according to the SlicedEllpack organization.
 */
template<
   typename SegmentsConstView,
   typename ArrayView,
   typename Index,
   typename Function,
   ElementsOrganization Organization,
   int SliceSize,
   int SegmentsPerBlock,
   int BlockSize = 256 >
__global__
void
forElements_SlicedEllpack_PerElement_WithIndexes(
   const Index gridIdx,
   const SegmentsConstView segments,
   const ArrayView segmentIndexes,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using CudaScan = Algorithms::detail::CudaBlockScanShfl< Algorithms::detail::ScanType::Exclusive, BlockSize, Plus, Index >;
   using ScanStorage = typename CudaScan::Storage;

   __shared__ ScanStorage scan_storage;
   __shared__ Index shared_offsets[ SegmentsPerBlock + 1 ];
   __shared__ Index shared_global_offsets[ SegmentsPerBlock ];
   __shared__ Index shared_segment_indexes[ SegmentsPerBlock ];

   const Index segmentIdx_ptr = Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock + threadIdx.x;
   const Index last_local_segment_idx =
      min( SegmentsPerBlock, segmentIndexes.getSize() - Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock );
   if( segmentIdx_ptr < segmentIndexes.getSize() && threadIdx.x < SegmentsPerBlock ) {
      TNL_ASSERT_LT( segmentIdx_ptr, segmentIndexes.getSize(), "" );
      const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
      shared_segment_indexes[ threadIdx.x ] = seg_idx;
      TNL_ASSERT_GE( shared_segment_indexes[ threadIdx.x ], 0, "" );
      TNL_ASSERT_LT( shared_segment_indexes[ threadIdx.x ], segments.getSegmentCount(), "" );
      shared_global_offsets[ threadIdx.x ] = segments.getGlobalIndex( seg_idx, 0 );
   }

   Index value = 0;
   if( segmentIdx_ptr < segmentIndexes.getSize() && threadIdx.x <= SegmentsPerBlock ) {
      const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
      TNL_ASSERT_GE( seg_idx, 0, "Wrong index of segment index - smaller that 0." );
      TNL_ASSERT_LT( seg_idx, segments.getSegmentCount(), "Wrong index of segment index - larger that the number of indexes." );
      value = segments.getSegmentSize( seg_idx );
   }
   const Index v = CudaScan::scan( Plus{}, (Index) 0, value, threadIdx.x, scan_storage );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      shared_offsets[ threadIdx.x ] = v;

   // Compute the last offset in the block - this is necessary only SegmentsPerBlock == BlockSize
   if constexpr( SegmentsPerBlock == BlockSize )
      if( threadIdx.x == last_local_segment_idx - 1 ) {
         TNL_ASSERT_LT( segmentIdx_ptr, segmentIndexes.getSize(), "" );
         const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
         TNL_ASSERT_GE( seg_idx, 0, "Wrong index of segment index - smaller that 0." );
         TNL_ASSERT_LT(
            seg_idx, segments.getSegmentCount(), "Wrong index of segment index - larger that the number of indexes." );
         shared_offsets[ threadIdx.x + 1 ] = shared_offsets[ threadIdx.x ] + segments.getSegmentSize( seg_idx );
      }
   __syncthreads();

   const Index last_idx = shared_offsets[ last_local_segment_idx ];
   TNL_ASSERT_LE( last_idx, segments.getStorageSize() - shared_offsets[ 0 ], "" );

   Index idx = threadIdx.x;
   while( idx < last_idx ) {
      auto [ found, local_segmentIdx ] = Algorithms::findUpperBound( shared_offsets, last_local_segment_idx + 1, idx );
      if( found ) {
         local_segmentIdx--;
         TNL_ASSERT_GE( local_segmentIdx, 0, "" );
         TNL_ASSERT_LT( local_segmentIdx, last_local_segment_idx, "" );
         TNL_ASSERT_LT( local_segmentIdx, SegmentsPerBlock, "" );
         TNL_ASSERT_GE( shared_segment_indexes[ local_segmentIdx ], 0, "" );
         TNL_ASSERT_LT( shared_segment_indexes[ local_segmentIdx ], segments.getSegmentCount(), "" );

         if constexpr( Organization == RowMajorOrder ) {
            const Index localIdx = idx - shared_offsets[ local_segmentIdx ];
            const Index globalIdx = shared_global_offsets[ local_segmentIdx ] + localIdx;
            TNL_ASSERT_GE( globalIdx, 0, "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            if constexpr( callableArgumentCount< Function >() == 3 )
               function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
            else
               function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
         }
         else {  // ColumnMajorOrder
            const Index localIdx = idx - shared_offsets[ local_segmentIdx ];
            const Index globalIdx = shared_global_offsets[ local_segmentIdx ] + localIdx * SliceSize;
            TNL_ASSERT_GE( globalIdx, 0, "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            if constexpr( callableArgumentCount< Function >() == 3 )
               function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
            else
               function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
         }
      }
      idx += BlockSize;
   }
#endif
}

/**
 * \brief Kernel for the PerElement thread mapping, traversing the segments of a contiguous range
 * that satisfy \e condition.
 *
 * Every block evaluates \e condition for \e SegmentsPerBlock consecutive segments of the range
 * [\e begin, \e end). The segments satisfying the condition are compacted using a block-wide
 * inclusive scan of the condition values and their local offsets are then computed by an exclusive
 * scan of their sizes. All elements of the active segments are divided evenly among the threads of
 * the block and each thread finds the segment its element belongs to by a binary search.
 *
 * This is the same algorithm as \ref forElements_CSR_PerElement_If, only the global index of an
 * element is computed according to the SlicedEllpack organization.
 */
template<
   typename SegmentsConstView,
   typename Index,
   typename Condition,
   typename Function,
   ElementsOrganization Organization,
   int SliceSize,
   int SegmentsPerBlock,
   int BlockSize = 256 >
__global__
void
forElements_SlicedEllpack_PerElement_If(
   const Index gridIdx,
   const SegmentsConstView segments,
   const Index begin,
   const Index end,
   Condition condition,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using InclusiveCudaScan =
      Algorithms::detail::CudaBlockScanShfl< Algorithms::detail::ScanType::Inclusive, BlockSize, Plus, Index >;
   using ExclusiveCudaScan =
      Algorithms::detail::CudaBlockScanShfl< Algorithms::detail::ScanType::Exclusive, BlockSize, Plus, Index >;
   using InclusiveScanStorage = typename InclusiveCudaScan::Storage;  // TODO: Storage should not depend on the scan type
   using ExclusiveScanStorage = typename ExclusiveCudaScan::Storage;

   __shared__ InclusiveScanStorage inclusive_scan_storage;
   __shared__ ExclusiveScanStorage exclusive_scan_storage;
   __shared__ Index conditions[ SegmentsPerBlock ];
   __shared__ Index shared_offsets[ SegmentsPerBlock + 1 ];
   __shared__ Index shared_global_offsets[ SegmentsPerBlock ];
   __shared__ Index shared_segment_indexes[ SegmentsPerBlock ];

   const Index segmentIdx = begin + Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock + threadIdx.x;
   const Index last_local_segment_idx =
      min( SegmentsPerBlock, end - begin - Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock );
   Index conditionValue = 0;
   if( segmentIdx < end && threadIdx.x < SegmentsPerBlock )
      conditionValue = condition( segmentIdx );

   // initialize all shared slots that any later phase may read: threads whose segment index
   // is out of range do not write anything below and their slots must still have a defined value
   if( threadIdx.x < SegmentsPerBlock ) {
      conditions[ threadIdx.x ] = 0;
      shared_offsets[ threadIdx.x ] = 0;
      shared_global_offsets[ threadIdx.x ] = 0;
      shared_segment_indexes[ threadIdx.x ] = 0;
   }
   if( threadIdx.x == 0 )
      shared_offsets[ SegmentsPerBlock ] = 0;
   __syncthreads();

   const Index v1 = InclusiveCudaScan::scan( Plus{}, (Index) 0, conditionValue, threadIdx.x, inclusive_scan_storage );
   if( threadIdx.x < SegmentsPerBlock && threadIdx.x < BlockSize )
      conditions[ threadIdx.x ] = v1;
   __syncthreads();

   __shared__ Index activeSegmentsCount;
   if( threadIdx.x == 0 )
      activeSegmentsCount = conditions[ SegmentsPerBlock - 1 ];
   const Index prevConditions = ( threadIdx.x == 0 || threadIdx.x > SegmentsPerBlock ) ? 0 : conditions[ threadIdx.x - 1 ];
   if( threadIdx.x < SegmentsPerBlock && conditions[ threadIdx.x ] != prevConditions ) {
      shared_segment_indexes[ conditions[ threadIdx.x ] - 1 ] = segmentIdx;
      shared_global_offsets[ conditions[ threadIdx.x ] - 1 ] =
         segments.getGlobalIndex( segmentIdx, 0 );  // TODO: get this using sliceIdx
      shared_offsets[ conditions[ threadIdx.x ] - 1 ] = segments.getSegmentSize( segmentIdx );
   }
   __syncthreads();

   // the threads beyond the segments of the block contribute nothing to the scan
   const Index segmentSize = ( threadIdx.x <= SegmentsPerBlock ) ? shared_offsets[ threadIdx.x ] : 0;
   const Index v2 = ExclusiveCudaScan::scan( Plus{}, (Index) 0, segmentSize, threadIdx.x, exclusive_scan_storage );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      shared_offsets[ threadIdx.x ] = v2;

   // Compute the last offset in the block - this is necessary only SegmentsPerBlock == BlockSize
   if constexpr( SegmentsPerBlock == BlockSize )
      if( threadIdx.x == last_local_segment_idx - 1 && activeSegmentsCount == BlockSize ) {
         const Index seg_idx = shared_segment_indexes[ threadIdx.x ];
         TNL_ASSERT_GE( seg_idx, 0, "Wrong index of segment index - smaller that 0." );
         TNL_ASSERT_LT(
            seg_idx, segments.getSegmentCount() - 1, "Wrong index of segment index - larger that the number of indexes." );
         shared_offsets[ threadIdx.x + 1 ] = shared_offsets[ threadIdx.x ] + segments.getSegmentSize( seg_idx );
      }
   __syncthreads();

   const Index last_idx = shared_offsets[ activeSegmentsCount ];
   if( activeSegmentsCount > 0 )
      TNL_ASSERT_LE( last_idx, segments.getStorageSize() - shared_segment_indexes[ 0 ], "" );

   Index idx = threadIdx.x;
   while( idx < last_idx ) {
      auto [ found, local_segmentIdx ] = Algorithms::findUpperBound( shared_offsets, activeSegmentsCount + 1, idx );
      if( found ) {
         local_segmentIdx--;
         TNL_ASSERT_GE( local_segmentIdx, 0, "" );
         TNL_ASSERT_LT( local_segmentIdx, activeSegmentsCount, "" );
         TNL_ASSERT_LT( local_segmentIdx, SegmentsPerBlock, "" );
         TNL_ASSERT_GE( shared_segment_indexes[ local_segmentIdx ], 0, "" );
         TNL_ASSERT_LT( shared_segment_indexes[ local_segmentIdx ], segments.getSegmentCount(), "" );

         if constexpr( Organization == RowMajorOrder ) {
            const Index localIdx = idx - shared_offsets[ local_segmentIdx ];
            const Index globalIdx = shared_global_offsets[ local_segmentIdx ] + localIdx;
            TNL_ASSERT_GE( globalIdx, 0, "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            if constexpr( callableArgumentCount< Function >() == 3 )
               function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
            else
               function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
         }
         else {  // ColumnMajorOrder
            const Index localIdx = idx - shared_offsets[ local_segmentIdx ];
            const Index globalIdx = shared_global_offsets[ local_segmentIdx ] + localIdx * SliceSize;
            TNL_ASSERT_GE( globalIdx, 0, "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            if constexpr( callableArgumentCount< Function >() == 3 )
               function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
            else
               function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
         }
      }
      idx += BlockSize;
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
