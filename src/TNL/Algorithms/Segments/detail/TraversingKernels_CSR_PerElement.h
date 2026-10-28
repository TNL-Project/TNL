// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/detail/CudaScanKernel.h>
#include <TNL/Algorithms/find.h>
#include <TNL/Assert.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Kernel for the PerElement thread mapping, traversing a contiguous range of segments.
 *
 * Unlike the other thread mappings, the threads are not assigned to segments but to elements: all
 * elements of the segments processed by a block are merged into one contiguous range, which is
 * divided evenly among the threads of the block regardless of the segment boundaries. Each thread
 * then finds the segment its element belongs to by a binary search in the offsets of the block's
 * segments stored in shared memory. The work is thus balanced perfectly within each block, but not
 * across the blocks, since every block processes a fixed number of segments.
 *
 * Every block processes `SegmentsPerBlock` consecutive segments of the range [`begin`, `end`);
 * their offsets are read directly from `offsets` into shared memory.
 *
 * This kind of work distribution is called nonzero splitting (as opposed to row splitting) in
 *
 * D. Merrill and M. Garland, "Merge-based parallel sparse matrix-vector multiplication," SC '16: Proceedings of the
 * International Conference for High Performance Computing, Networking, Storage and Analysis, Salt Lake City, UT, USA,
 * 2016, pp. 678-689.
 *
 * which balances the elements globally using the merge-path decomposition; here the balancing is
 * restricted to the block and a plain binary search is used instead.
 */
template< typename OffsetsView, typename Index, typename Function, int SegmentsPerBlock = 256, int BlockSize = 256 >
__global__
void
forElements_CSR_PerElement(
   const Index gridIdx,
   const OffsetsView offsets,
   const Index begin,
   const Index end,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   static_assert( SegmentsPerBlock <= BlockSize, "Every segment of the block must have its own thread." );

   __shared__ Index shared_offsets[ SegmentsPerBlock + 1 ];
   const Index first_segment_in_block = begin + Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock;
   const Index last_segment_in_block = min( end, first_segment_in_block + SegmentsPerBlock );
   const Index segments_in_block = last_segment_in_block - first_segment_in_block;
   if( segments_in_block <= 0 )
      return;
   if( threadIdx.x < segments_in_block )
      shared_offsets[ threadIdx.x ] = offsets[ first_segment_in_block + threadIdx.x ];
   if( threadIdx.x == 0 )
      shared_offsets[ segments_in_block ] = offsets[ last_segment_in_block ];
   __syncthreads();

   const Index first_idx = shared_offsets[ 0 ];
   const Index last_idx = shared_offsets[ segments_in_block ];

   Index idx = threadIdx.x;
   while( idx + first_idx < last_idx ) {
      auto [ found, local_segmentIdx ] = Algorithms::findUpperBound( shared_offsets, segments_in_block + 1, idx + first_idx );
      if( found ) {
         local_segmentIdx--;
         TNL_ASSERT_LT( first_idx + idx, last_idx, "" );
         const Index globalIdx = first_idx + idx;
         if constexpr( callableArgumentCount< Function >() == 3 )
            function( first_segment_in_block + local_segmentIdx, globalIdx - shared_offsets[ local_segmentIdx ], globalIdx );
         else
            function( first_segment_in_block + local_segmentIdx, globalIdx );
      }
      idx += BlockSize;
   }
#endif
}

/**
 * \brief Kernel for the PerElement thread mapping, traversing the segments given explicitly by
 * \e segmentIndexes.
 *
 * Unlike the other thread mappings, the threads are not assigned to segments but to elements: all
 * elements of the segments processed by a block are merged into one contiguous range, which is
 * divided evenly among the threads of the block regardless of the segment boundaries. Each thread
 * then finds the segment its element belongs to by a binary search in the offsets of the block's
 * segments stored in shared memory. The work is thus balanced perfectly within each block, but not
 * across the blocks, since every block processes a fixed number of segments.
 *
 * Every block processes \e SegmentsPerBlock consecutive entries of \e segmentIndexes. Since the
 * segments are not contiguous in memory, their local offsets within the block are computed by a
 * block-wide exclusive scan of their sizes and their global offsets are stored separately. See
 * \ref forElements_CSR_PerElement for the literature reference.
 */
template< typename OffsetsView, typename ArrayView, typename Index, typename Function, int SegmentsPerBlock, int BlockSize = 256 >
__global__
void
forElements_CSR_PerElement_WithIndexes(
   const Index gridIdx,
   const OffsetsView offsets,
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
      shared_segment_indexes[ threadIdx.x ] = segmentIndexes[ segmentIdx_ptr ];
      TNL_ASSERT_GE( shared_segment_indexes[ threadIdx.x ], 0, "" );
      TNL_ASSERT_LT( shared_segment_indexes[ threadIdx.x ], offsets.getSize(), "" );
      shared_global_offsets[ threadIdx.x ] = offsets[ shared_segment_indexes[ threadIdx.x ] ];
   }

   Index value = 0;
   if( segmentIdx_ptr < segmentIndexes.getSize() && threadIdx.x <= SegmentsPerBlock ) {
      const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
      TNL_ASSERT_GE( seg_idx, 0, "Wrong index of segment index - smaller that 0." );
      TNL_ASSERT_LT( seg_idx, offsets.getSize() - 1, "Wrong index of segment index - larger that the number of indexes." );
      value = offsets[ seg_idx + 1 ] - offsets[ seg_idx ];
   }
   const Index v = CudaScan::scan( Plus{}, (Index) 0, value, threadIdx.x, scan_storage );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      shared_offsets[ threadIdx.x ] = v;

   // Compute the last offset in the block - this is necessary only SegmentsPerBlock == BlockSize
   if constexpr( SegmentsPerBlock == BlockSize ) {
      __syncthreads();
      if( threadIdx.x == last_local_segment_idx - 1 ) {
         const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
         TNL_ASSERT_GE( seg_idx, 0, "Wrong index of segment index - smaller that 0." );
         TNL_ASSERT_LT( seg_idx, offsets.getSize() - 1, "Wrong index of segment index - larger that the number of indexes." );
         shared_offsets[ threadIdx.x + 1 ] = shared_offsets[ threadIdx.x ] + offsets[ seg_idx + 1 ] - offsets[ seg_idx ];
      }
   }

   __syncthreads();
   const Index last_idx = shared_offsets[ last_local_segment_idx ];

   Index idx = threadIdx.x;
   while( idx < last_idx ) {
      auto [ found, local_segmentIdx ] = Algorithms::findUpperBound( shared_offsets, last_local_segment_idx + 1, idx );
      if( found ) {
         local_segmentIdx--;
         TNL_ASSERT_GE( local_segmentIdx, 0, "" );
         TNL_ASSERT_LT( local_segmentIdx, last_local_segment_idx, "" );
         TNL_ASSERT_LT( local_segmentIdx, SegmentsPerBlock, "" );
         TNL_ASSERT_GE( shared_segment_indexes[ local_segmentIdx ], 0, "" );
         TNL_ASSERT_LT( shared_segment_indexes[ local_segmentIdx ], offsets.getSize() - 1, "" );

         const Index localIdx = idx - shared_offsets[ local_segmentIdx ];
         const Index globalIdx = shared_global_offsets[ local_segmentIdx ] + localIdx;
         TNL_ASSERT_GE( globalIdx, 0, "" );
         TNL_ASSERT_LT( globalIdx, offsets[ offsets.getSize() - 1 ], "" );
         if constexpr( callableArgumentCount< Function >() == 3 )
            function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
         else
            function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
      }
      idx += BlockSize;
   }

#endif
}

/**
 * \brief Kernel for the PerElement thread mapping, traversing the segments of a contiguous range
 * that satisfy \e condition.
 *
 * Unlike the other thread mappings, the threads are not assigned to segments but to elements: all
 * elements of the segments processed by a block are merged into one contiguous range, which is
 * divided evenly among the threads of the block regardless of the segment boundaries. Each thread
 * then finds the segment its element belongs to by a binary search in the offsets of the block's
 * segments stored in shared memory. The work is thus balanced perfectly within each block, but not
 * across the blocks, since every block processes a fixed number of segments.
 *
 * Every block evaluates \e condition for \e SegmentsPerBlock consecutive segments of the range
 * [\e begin, \e end). The segments satisfying the condition are compacted using a block-wide
 * inclusive scan of the condition values and their local offsets are then computed by an exclusive
 * scan of their sizes, so only the elements of the active segments are divided among the threads.
 * See \ref forElements_CSR_PerElement for the literature reference.
 */
template< typename OffsetsView, typename Index, typename Condition, typename Function, int SegmentsPerBlock, int BlockSize = 256 >
__global__
void
forElements_CSR_PerElement_If(
   Index gridIdx,
   OffsetsView offsets,
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
   using ScanStorage = typename InclusiveCudaScan::Storage;  // TODO: Storage should not depend on the scan type

   __shared__ ScanStorage scan_storage;
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

   const Index v1 = InclusiveCudaScan::scan( Plus{}, (Index) 0, conditionValue, threadIdx.x, scan_storage );
   if( threadIdx.x < SegmentsPerBlock && threadIdx.x < BlockSize )
      conditions[ threadIdx.x ] = v1;
   __syncthreads();

   __shared__ Index activeSegmentsCount;
   if( threadIdx.x == 0 )
      activeSegmentsCount = conditions[ SegmentsPerBlock - 1 ];
   const Index prevConditions = ( threadIdx.x == 0 || threadIdx.x > SegmentsPerBlock ) ? 0 : conditions[ threadIdx.x - 1 ];
   if( threadIdx.x < SegmentsPerBlock && conditions[ threadIdx.x ] != prevConditions ) {
      TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
      shared_segment_indexes[ conditions[ threadIdx.x ] - 1 ] = segmentIdx;
      shared_global_offsets[ conditions[ threadIdx.x ] - 1 ] = offsets[ segmentIdx ];
      shared_offsets[ conditions[ threadIdx.x ] - 1 ] = offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
   }
   __syncthreads();

   // the threads beyond the segments of the block contribute nothing to the scan
   const Index segmentSize = ( threadIdx.x <= SegmentsPerBlock ) ? shared_offsets[ threadIdx.x ] : 0;
   const Index v2 = ExclusiveCudaScan::scan( Plus{}, (Index) 0, segmentSize, threadIdx.x, scan_storage );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      shared_offsets[ threadIdx.x ] = v2;

   // Compute the last offset in the block - this is necessary only SegmentsPerBlock == BlockSize
   if constexpr( SegmentsPerBlock == BlockSize ) {
      __syncthreads();
      if( threadIdx.x == last_local_segment_idx - 1 && activeSegmentsCount == BlockSize ) {
         const Index seg_idx = shared_segment_indexes[ threadIdx.x ];
         TNL_ASSERT_GE( seg_idx, 0, "Wrong index of segment index - smaller that 0." );
         TNL_ASSERT_LT( seg_idx, offsets.getSize() - 1, "Wrong index of segment index - larger that the number of indexes." );
         shared_offsets[ threadIdx.x + 1 ] = shared_offsets[ threadIdx.x ] + offsets[ seg_idx + 1 ] - offsets[ seg_idx ];
      }
   }
   __syncthreads();

   const Index last_idx = shared_offsets[ activeSegmentsCount ];
   if( activeSegmentsCount > 0 )
      TNL_ASSERT_LE( last_idx, offsets[ offsets.getSize() - 1 ] - offsets[ shared_segment_indexes[ 0 ] ], "" );

   Index idx = threadIdx.x;
   while( idx < last_idx ) {
      auto [ found, local_segmentIdx ] = Algorithms::findUpperBound( shared_offsets, activeSegmentsCount + 1, idx );
      if( found ) {
         local_segmentIdx--;
         TNL_ASSERT_GE( local_segmentIdx, 0, "" );
         TNL_ASSERT_LT( local_segmentIdx, activeSegmentsCount, "" );
         TNL_ASSERT_LT( local_segmentIdx, SegmentsPerBlock, "" );
         TNL_ASSERT_GE( shared_segment_indexes[ local_segmentIdx ], 0, "" );
         TNL_ASSERT_LT( shared_segment_indexes[ local_segmentIdx ], offsets.getSize() - 1, "" );

         const Index localIdx = idx - shared_offsets[ local_segmentIdx ];
         const Index globalIdx = shared_global_offsets[ local_segmentIdx ] + localIdx;

         TNL_ASSERT_GE( globalIdx, 0, "" );
         TNL_ASSERT_LT( globalIdx, offsets[ offsets.getSize() - 1 ], "" );
         if constexpr( callableArgumentCount< Function >() == 3 )
            function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
         else
            function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
      }
      idx += BlockSize;
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
