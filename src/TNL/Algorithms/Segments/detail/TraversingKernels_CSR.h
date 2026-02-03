// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/AtomicOperations.h>

namespace TNL::Algorithms::Segments::detail {

template< typename OffsetsView, typename Index, typename Function, int BlockSize = 256 >
__device__
void
dynamicGroupingTraversingKernel_CSR( const OffsetsView offsets,
                                     bool traverse_segment,
                                     const Index segmentIdx,
                                     Function function )
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

      Index globalIdx = offsets[ scheduled_segment[ 0 ] ];
      const Index endIdx = offsets[ scheduled_segment[ 0 ] + 1 ];

      if constexpr( argumentCount< Function >() == 3 ) {
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
   if( traverse_segment && segment_size > warpSize )
      warps_scheduler[ AtomicOperations< Devices::GPU >::add( active_warps[ 0 ], 1 ) ] = segmentIdx;
   __syncthreads();

   // Now traverse scheduled segments in warps
   Index warp_idx = threadIdx.x / warpSize;
   while( warp_idx < active_warps[ 0 ] ) {
      Index scheduled_segment = warps_scheduler[ warp_idx ];
      Index globalIdx = offsets[ scheduled_segment ] + ( threadIdx.x & ( warpSize - 1 ) );  // & is cheaper than %
      const Index endIdx = offsets[ scheduled_segment + 1 ];
      if constexpr( argumentCount< Function >() == 3 ) {
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
      if( segmentIdx == scheduled_segment ) {
         traverse_segment = false;
      }
      warp_idx += warpsPerBlock;
   }

   // Processing segments smaller than or equal to warp size
   if( traverse_segment ) {
      Index globalIdx = offsets[ segmentIdx ];
      const Index endIdx = offsets[ segmentIdx + 1 ];
      if constexpr( argumentCount< Function >() == 3 ) {
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

template< typename OffsetsView, typename Index, typename Function, int BlockSize = 256 >
__global__
void
forElementsDynamicGroupingKernel_CSR( const Index gridIdx,
                                      const OffsetsView offsets,
                                      const Index begin,
                                      const Index end,
                                      Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx );
   bool traverse_segment = ( segmentIdx < end );

   dynamicGroupingTraversingKernel_CSR( offsets, traverse_segment, segmentIdx, function );
#endif
}

template< typename OffsetsView, typename Index, typename Function, int BlockSize = 256 >
__global__
void
forElementsBlockMergeKernel_CSR( const Index gridIdx,
                                 const OffsetsView offsets,
                                 const Index begin,
                                 const Index end,
                                 Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   __shared__ Index shared_offsets[ BlockSize + 1 ];
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx );
   if( segmentIdx <= end )
      shared_offsets[ threadIdx.x ] = offsets[ segmentIdx ];
   if( threadIdx.x == 0 && end - begin >= BlockSize )
      shared_offsets[ BlockSize ] = offsets[ end ];
   __syncthreads();

   const Index first_segment_in_block = segmentIdx - threadIdx.x;
   const Index last_segment_in_block = min( end, first_segment_in_block + BlockSize );
   const Index segments_in_block = last_segment_in_block - first_segment_in_block;
   const Index first_idx = shared_offsets[ 0 ];
   const Index last_idx = offsets[ last_segment_in_block ];

   Index idx = threadIdx.x;
   while( idx + first_idx < last_idx ) {
      auto [ found, local_segmentIdx ] = Algorithms::findUpperBound( shared_offsets, segments_in_block + 1, idx + first_idx );
      if( found ) {
         local_segmentIdx--;
         TNL_ASSERT_LT( first_idx + idx, last_idx, "" );
         const Index globalIdx = first_idx + idx;
         if constexpr( argumentCount< Function >() == 3 )
            function( first_segment_in_block + local_segmentIdx, globalIdx - shared_offsets[ local_segmentIdx ], globalIdx );
         else
            function( first_segment_in_block + local_segmentIdx, globalIdx );
      }
      idx += BlockSize;
   }
#endif
}

template< typename OffsetsView, typename Index, typename Function >
__global__
void
forElementsKernel_CSR( const Index gridIdx,
                       const Index threadsPerSegment,
                       OffsetsView offsets,
                       const Index begin,
                       const Index end,
                       Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
   if( segmentIdx >= end )
      return;

   const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   for( Index globalIdx = offsets[ segmentIdx ] + laneIdx; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      if constexpr( argumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
      localIdx += threadsPerSegment;
   }
#endif
}

template< typename OffsetsView, typename ArrayView, typename Index, typename Function >
__global__
void
forElementsWithSegmentIndexesKernel_CSR( const Index gridIdx,
                                         const Index threadsPerSegment,
                                         const OffsetsView offsets,
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
   TNL_ASSERT_LT( segmentIdx, offsets.getSize() - 1, "Wrong index segment index - larger that the number of indexes." );

   const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   for( Index globalIdx = offsets[ segmentIdx ] + laneIdx; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      if constexpr( argumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
      localIdx += threadsPerSegment;
   }
#endif
}

template< typename OffsetsView, typename ArrayView, typename Index, typename Function, int BlockSize = 256 >
__global__
void
forElementsWithSegmentIndexesDynamicGroupingKernel_CSR( const Index gridIdx,
                                                        const OffsetsView offsets,
                                                        const ArrayView segmentIndexes,
                                                        Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   const Index segmentIdx_ptr = Backend::getGlobalThreadIdx_x( gridIdx );

   Index segmentIdx( 0 );
   bool traverse_segment( false );
   if( segmentIdx_ptr < segmentIndexes.getSize() ) {
      TNL_ASSERT_GE( segmentIdx_ptr, 0, "" );
      TNL_ASSERT_LT( segmentIdx_ptr, segmentIndexes.getSize(), "" );
      segmentIdx = segmentIndexes[ segmentIdx_ptr ];
      TNL_ASSERT_GE( segmentIdx, 0, "Wrong index of segment index - smaller that 0." );
      TNL_ASSERT_LT( segmentIdx, offsets.getSize() - 1, "Wrong index of segment index - larger that the number of indexes." );
      traverse_segment = true;
   }

   dynamicGroupingTraversingKernel_CSR( offsets, traverse_segment, segmentIdx, function );
#endif
}

template< typename OffsetsView, typename ArrayView, typename Index, typename Function, int SegmentsPerBlock, int BlockSize = 256 >
__global__
void
forElementsWithSegmentIndexesBlockMergeKernel_CSR( const Index gridIdx,
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
   TNL_ASSERT_LE( last_idx, offsets[ offsets.getSize() - 1 ] - offsets[ shared_segment_indexes[ 0 ] ], "" );

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
         if constexpr( argumentCount< Function >() == 3 )
            function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
         else
            function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
      }
      idx += BlockSize;
   }

#endif
}

template< typename OffsetsView, typename Index, typename Condition, typename Function >
__global__
void
forElementsIfKernel_CSR( const Index gridIdx,
                         const Index threadsPerSegment,
                         const OffsetsView offsets,
                         const Index begin,
                         const Index end,
                         Condition condition,
                         Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
   if( segmentIdx >= end || ! condition( segmentIdx ) )
      return;

   const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   for( Index globalIdx = offsets[ segmentIdx ] + laneIdx; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      if constexpr( argumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
      localIdx += threadsPerSegment;
   }
#endif
}

template< typename OffsetsView, typename Index, typename Condition, typename Function, int BlockSize = 256 >
__global__
void
forElementsIfDynamicGroupingKernel_CSR( const Index gridIdx,
                                        const OffsetsView offsets,
                                        const Index begin,
                                        const Index end,
                                        Condition condition,
                                        Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx );

   bool traverse_segment = false;
   if( segmentIdx < end ) {
      traverse_segment = condition( segmentIdx );
   }

   dynamicGroupingTraversingKernel_CSR( offsets, traverse_segment, segmentIdx, function );
#endif
}

template< typename OffsetsView, typename Index, typename Condition, typename Function, int SegmentsPerBlock, int BlockSize = 256 >
__global__
void
forElementsIfBlockMergeKernel_CSR( Index gridIdx,
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
   const Index last_local_segment_idx = min( SegmentsPerBlock, end - begin - blockIdx.x * SegmentsPerBlock );
   Index conditionValue = 0;
   if( segmentIdx < end && threadIdx.x < SegmentsPerBlock ) {
      conditionValue = condition( segmentIdx );
      shared_offsets[ threadIdx.x ] = 0;
   }
   __syncthreads();

   const Index v1 = InclusiveCudaScan::scan( Plus{}, (Index) 0, conditionValue, threadIdx.x, scan_storage );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      conditions[ threadIdx.x ] = v1;
   __syncthreads();

   __shared__ Index activeSegmentsCount;
   if( threadIdx.x == 0 )
      activeSegmentsCount = conditions[ SegmentsPerBlock - 1 ];
   if( ( threadIdx.x == 0 && conditions[ 0 ] != 0 ) || conditions[ threadIdx.x ] != conditions[ threadIdx.x - 1 ] ) {
      TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
      shared_segment_indexes[ conditions[ threadIdx.x ] - 1 ] = segmentIdx;
      shared_global_offsets[ conditions[ threadIdx.x ] - 1 ] = offsets[ segmentIdx ];
      shared_offsets[ conditions[ threadIdx.x ] - 1 ] = offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
   }
   __syncthreads();

   Index segmentSize = shared_offsets[ threadIdx.x ];
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
         if constexpr( argumentCount< Function >() == 3 )
            function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
         else
            function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
      }
      idx += BlockSize;
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
