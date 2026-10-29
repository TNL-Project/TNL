// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Assert.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Device function implementing the Tiered thread mapping
 * for the traversal of CSR segments.
 *
 * Every thread of the block passes the segment it owns (`segmentIdx`,
 * valid only if `traverse_segment` is true).
 * The owned segments are then traversed in three tiers according to their size:
 *
 * 1. Segments larger than the block are traversed one
 *    at a time by the whole block.
 *    The next such segment is picked via an atomic CAS on a shared variable.
 * 2. Segments larger than a warp but not larger than the block are
 *    registered in a shared queue via an atomic counter and the warps of the
 *    block take them from the queue one by one.
 * 3. Segments not larger than a warp are traversed
 *    sequentially by the thread that owns them.
 *
 * See \ref reduceSegments_CSR_Tiered for the reducing
 * counterpart and the literature reference.
 */
template< typename OffsetsView, typename Index, typename Function, int BlockSize = 256 >
__device__
void
forElements_CSR_Tiered_impl( const OffsetsView offsets, bool traverse_segment, const Index segmentIdx, Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   constexpr Index warpSize = Backend::getWarpSize();
   constexpr Index warpsPerBlock = BlockSize / warpSize;
   __shared__ Index warps_scheduler[ BlockSize ];

   // Processing segments larger than BlockSize
   __shared__ Index scheduled_segment[ 1 ];
   constexpr Index none_scheduled = -1;
   Index segment_size = -1;
   if( traverse_segment ) {
      segment_size = offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
   }

   if( threadIdx.x == 0 )
      scheduled_segment[ 0 ] = none_scheduled;
   __syncthreads();
   while( true ) {
      if( traverse_segment && segment_size > BlockSize ) {
         AtomicOperations< Devices::GPU >::CAS( scheduled_segment[ 0 ], scheduled_segment[ 0 ], segmentIdx );
      }
      __syncthreads();
      if( scheduled_segment[ 0 ] == none_scheduled )
         break;

      Index globalIdx = offsets[ scheduled_segment[ 0 ] ] + threadIdx.x;
      const Index endIdx = offsets[ scheduled_segment[ 0 ] + 1 ];

      if constexpr( callableArgumentCount< Function >() == 3 ) {
         Index localIdx = threadIdx.x;
         while( globalIdx < endIdx ) {
            function( scheduled_segment[ 0 ], localIdx, globalIdx );
            localIdx += BlockSize;
            globalIdx += BlockSize;
         }
      }
      else
         while( globalIdx < endIdx ) {
            function( scheduled_segment[ 0 ], globalIdx );
            globalIdx += BlockSize;
         }
      // All threads must finish reading scheduled_segment before it is reset
      __syncthreads();

      // Mark segment as processed
      if( segmentIdx == scheduled_segment[ 0 ] ) {
         traverse_segment = false;
         scheduled_segment[ 0 ] = none_scheduled;
      }
      __syncthreads();
   }

   // Processing segments smaller than BlockSize and larger the warp size
   __shared__ int active_warps[ 1 ];
   if( threadIdx.x == 0 )
      active_warps[ 0 ] = 0;
   __syncthreads();

   // Each thread owning segment with size larger than warpSize registers for scheduling
   if( traverse_segment && segment_size > warpSize ) {
      warps_scheduler[ AtomicOperations< Devices::GPU >::add( active_warps[ 0 ], 1 ) ] = segmentIdx;
      traverse_segment = false;
   }
   __syncthreads();

   // Now traverse scheduled segments in warps
   Index warp_idx = threadIdx.x / warpSize;
   while( warp_idx < active_warps[ 0 ] ) {
      Index scheduled_segment = warps_scheduler[ warp_idx ];
      Index globalIdx = offsets[ scheduled_segment ] + ( threadIdx.x & ( warpSize - 1 ) );  // & is cheaper than %
      const Index endIdx = offsets[ scheduled_segment + 1 ];
      if constexpr( callableArgumentCount< Function >() == 3 ) {
         Index localIdx = threadIdx.x & ( warpSize - 1 );  // & is cheaper than %
         for( ; globalIdx < endIdx; globalIdx += warpSize ) {
            function( scheduled_segment, localIdx, globalIdx );
            localIdx += warpSize;
         }
      }
      else {
         for( ; globalIdx < endIdx; globalIdx += warpSize ) {
            function( scheduled_segment, globalIdx );
         }
      }
      warp_idx += warpsPerBlock;
   }

   // Processing segments smaller than or equal to warp size
   if( traverse_segment ) {
      Index globalIdx = offsets[ segmentIdx ];
      const Index endIdx = offsets[ segmentIdx + 1 ];
      if constexpr( callableArgumentCount< Function >() == 3 ) {
         Index localIdx = 0;
         for( ; globalIdx < endIdx; globalIdx++ ) {
            function( segmentIdx, localIdx, globalIdx );
            localIdx++;
         }
      }
      else {
         for( ; globalIdx < endIdx; globalIdx++ ) {
            function( segmentIdx, globalIdx );
         }
      }
   }
#endif
}

/**
 * \brief Kernel for the Tiered thread mapping, traversing a
 * contiguous range of segments.
 *
 * Every block owns `BlockSize / threadsPerSegment` consecutive segments of the
 * range [`begin`, `end`), each of them owned by one thread, and the segments
 * are traversed by \ref forElements_CSR_Tiered_impl.
 * The remaining threads of the block own no segment and
 * only help with the larger tiers.
 */
template< typename OffsetsView, typename Index, typename Function, int BlockSize = 256 >
__global__
void
forElements_CSR_Tiered(
   const Index gridIdx,
   const Index threadsPerSegment,
   const OffsetsView offsets,
   const Index begin,
   const Index end,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentsPerBlock = BlockSize / threadsPerSegment;
   const Index segmentIdx = begin + Backend::getGlobalBlockIdx_x( gridIdx ) * segmentsPerBlock + threadIdx.x;
   bool traverse_segment = ( threadIdx.x < segmentsPerBlock && segmentIdx < end );

   forElements_CSR_Tiered_impl( offsets, traverse_segment, segmentIdx, function );
#endif
}

/**
 * \brief Kernel for the Tiered thread mapping, traversing the segments
 * given explicitly by `segmentIndexes`.
 *
 * Every block owns `BlockSize / threadsPerSegment` consecutive entries of
 * `segmentIndexes`, each of them owned by one thread, and the segments are
 * traversed by \ref forElements_CSR_Tiered_impl.
 */
template< typename OffsetsView, typename ArrayView, typename Index, typename Function, int BlockSize = 256 >
__global__
void
forElements_CSR_Tiered_WithIndexes(
   const Index gridIdx,
   const Index threadsPerSegment,
   const OffsetsView offsets,
   const ArrayView segmentIndexes,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentsPerBlock = BlockSize / threadsPerSegment;
   const Index segmentIdx_ptr = Backend::getGlobalBlockIdx_x( gridIdx ) * segmentsPerBlock + threadIdx.x;

   Index segmentIdx( 0 );
   bool traverse_segment( false );
   if( threadIdx.x < segmentsPerBlock && segmentIdx_ptr < segmentIndexes.getSize() ) {
      TNL_ASSERT_GE( segmentIdx_ptr, 0, "" );
      TNL_ASSERT_LT( segmentIdx_ptr, segmentIndexes.getSize(), "" );
      segmentIdx = segmentIndexes[ segmentIdx_ptr ];
      TNL_ASSERT_GE( segmentIdx, 0, "Wrong index of segment index - smaller that 0." );
      TNL_ASSERT_LT( segmentIdx, offsets.getSize() - 1, "Wrong index of segment index - larger that the number of indexes." );
      traverse_segment = true;
   }

   forElements_CSR_Tiered_impl( offsets, traverse_segment, segmentIdx, function );
#endif
}

/**
 * \brief Kernel for the Tiered thread mapping, traversing the segments of a
 * contiguous range that satisfy `condition`.
 *
 * Every block owns `BlockSize / threadsPerSegment` consecutive segments of
 * the range [`begin`, `end`), each of them owned by one thread which
 * evaluates `condition` for it; only the segments satisfying the condition
 * are passed to \ref forElements_CSR_Tiered_impl.
 */
template< typename OffsetsView, typename Index, typename Condition, typename Function, int BlockSize = 256 >
__global__
void
forElements_CSR_Tiered_If(
   const Index gridIdx,
   const Index threadsPerSegment,
   const OffsetsView offsets,
   const Index begin,
   const Index end,
   Condition condition,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentsPerBlock = BlockSize / threadsPerSegment;
   const Index segmentIdx = begin + Backend::getGlobalBlockIdx_x( gridIdx ) * segmentsPerBlock + threadIdx.x;

   bool traverse_segment = false;
   if( threadIdx.x < segmentsPerBlock && segmentIdx < end ) {
      traverse_segment = condition( segmentIdx );
   }

   forElements_CSR_Tiered_impl( offsets, traverse_segment, segmentIdx, function );
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
