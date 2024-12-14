// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

//#define USE_CUB
#ifdef __CUDACC__
   #include <cub/cub.cuh>
#endif

namespace TNL::Algorithms::Segments::detail {

template< typename OffsetsView, typename Index, typename Function, int BlockSize = 256 >
__global__
void
forElementsBlockMergeKernel( Index gridIdx, OffsetsView offsets, Index begin, Index end, Function function )
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
forElementsKernel( Index gridIdx, OffsetsView offsets, Index begin, Index end, Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   // We map one warp to each segment
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / Backend::getWarpSize();
   if( segmentIdx >= end )
      return;

   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   for( Index globalIdx = offsets[ segmentIdx ] + laneIdx; globalIdx < endIdx; globalIdx += Backend::getWarpSize() ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      if constexpr( argumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
      localIdx += Backend::getWarpSize();
   }
#endif
}

template< typename OffsetsView, typename ArrayView, typename Index, typename Function >
__global__
void
forElementsWithSegmentIndexesKernel( Index gridIdx,
                                     OffsetsView offsets,
                                     ArrayView segmentIndexes,
                                     Index begin,
                                     Index end,
                                     Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   // We map one warp to each segment
   const Index idx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / Backend::getWarpSize();
   if( idx >= end )
      return;
   TNL_ASSERT_GE( idx, 0, "" );
   TNL_ASSERT_LT( idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ idx ];
   TNL_ASSERT_GE( segmentIdx, 0, "Wrong index segment index - smaller that 0." );
   TNL_ASSERT_LT( segmentIdx, offsets.getSize() - 1, "Wrong index segment index - larger that the number of indexes." );

   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   for( Index globalIdx = offsets[ segmentIdx ] + laneIdx; globalIdx < endIdx; globalIdx += Backend::getWarpSize() ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      if constexpr( argumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
      localIdx += Backend::getWarpSize();
   }
#endif
}

template< typename OffsetsView, typename ArrayView, typename Index, typename Function, int SegmentsPerBlock, int BlockSize = 256 >
__global__
void
forElementsWithSegmentIndexesBlockMergeKernel( Index gridIdx,
                                               OffsetsView offsets,
                                               ArrayView segmentIndexes,
                                               const Index begin,
                                               const Index end,
                                               Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using CudaScan = Algorithms::detail::CudaBlockScanShfl< Algorithms::detail::ScanType::Exclusive, BlockSize, Plus, Index >;
   using ScanStorage = typename CudaScan::Storage;

   __shared__ ScanStorage scan_storage;
   __shared__ Index shared_offsets[ SegmentsPerBlock + 1 ];
   __shared__ Index shared_global_offsets[ SegmentsPerBlock ];
   __shared__ Index shared_segment_indexes[ SegmentsPerBlock ];

   const Index segmentIdx_ptr = begin + Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock + threadIdx.x;
   const Index last_local_segment_idx = min( SegmentsPerBlock, end - begin - blockIdx.x * SegmentsPerBlock );
   if( segmentIdx_ptr < end && threadIdx.x < SegmentsPerBlock ) {
      shared_segment_indexes[ threadIdx.x ] = segmentIndexes[ segmentIdx_ptr ];
      shared_global_offsets[ threadIdx.x ] = offsets[ shared_segment_indexes[ threadIdx.x ] ];
   }

   #ifdef USE_CUB
   using BlockScan = cub::BlockScan< Index, 256 >;
   __shared__ typename BlockScan::TempStorage temp_storage;
   Index value = 0;
   if( segmentIdx_ptr < end && threadIdx.x <= SegmentsPerBlock ) {
      const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
      value = offsets[ seg_idx + 1 ] - offsets[ seg_idx ];
   }
   BlockScan( temp_storage ).ExclusiveSum( value, value );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      shared_offsets[ threadIdx.x ] = value;
   #else  // USE_CUB
   Index value = 0;
   if( segmentIdx_ptr < end && threadIdx.x <= SegmentsPerBlock ) {
      const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
      TNL_ASSERT_GE( seg_idx, 0, "Wrong index of segment index - smaller that 0." );
      TNL_ASSERT_LT( seg_idx, offsets.getSize() - 1, "Wrong index of segment index - larger that the number of indexes." );
      value = offsets[ seg_idx + 1 ] - offsets[ seg_idx ];
   }
   const Index v = CudaScan::scan( Plus{}, (Index) 0, value, threadIdx.x, scan_storage );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      shared_offsets[ threadIdx.x ] = v;
   #endif

   // Compute the last offset in the block - this is necessary only SegmentsPerBlock == BlockSize
   if constexpr( SegmentsPerBlock == BlockSize )
      if( threadIdx.x == last_local_segment_idx - 1 ) {
         const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
         shared_offsets[ threadIdx.x + 1 ] = shared_offsets[ threadIdx.x ] + offsets[ seg_idx + 1 ] - offsets[ seg_idx ];
      }
   __syncthreads();
   const Index last_idx = shared_offsets[ last_local_segment_idx ];
   TNL_ASSERT_LT( last_idx, offsets[ offsets.getSize() - 1 ] - shared_segment_indexes[ 0 ], "" );

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
forElementsIfKernel( Index gridIdx, OffsetsView offsets, Index begin, Index end, Condition condition, Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   // We map one warp to each segment
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / Backend::getWarpSize();
   if( segmentIdx >= end || ! condition( segmentIdx ) )
      return;

   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   for( Index globalIdx = offsets[ segmentIdx ] + laneIdx; globalIdx < endIdx; globalIdx += Backend::getWarpSize() ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      function( segmentIdx, localIdx, globalIdx );
      localIdx += Backend::getWarpSize();
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail