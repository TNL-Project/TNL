// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

template< typename BlocksView,
          typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
reduceSegmentsCSRAdaptiveKernel( int gridIdx,
                                 BlocksView blocks,
                                 Offsets offsets,
                                 Fetch fetch,
                                 Reduction reduction,
                                 ResultKeeper keep,
                                 Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   using BlockType = detail::CSRAdaptiveKernelBlockDescriptor< Index >;
   constexpr int CudaBlockSize = detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::CudaBlockSize();
   constexpr int WarpSize = Backend::getWarpSize();
   constexpr int WarpsCount = detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::WarpsCount();
   constexpr std::size_t StreamedSharedElementsPerWarp =
      detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::StreamedSharedElementsPerWarp();

   __shared__ ReturnType streamShared[ WarpsCount ][ StreamedSharedElementsPerWarp ];
   __shared__ ReturnType multivectorShared[ CudaBlockSize / WarpSize ];
   //__shared__ BlockType sharedBlocks[ WarpsCount ];

   const Index index = ( ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x ) + threadIdx.x;
   const Index blockIdx = index / WarpSize;
   if( blockIdx >= blocks.getSize() - 1 )
      return;

   if( threadIdx.x < CudaBlockSize / WarpSize )
      multivectorShared[ threadIdx.x ] = identity;
   __syncthreads();
   ReturnType result = identity;
   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   /*if( laneIdx == 0 )
      sharedBlocks[ warpIdx ] = blocks[ blockIdx ];
   __syncthreads();
   const auto& block = sharedBlocks[ warpIdx ];*/
   const auto& block = blocks[ blockIdx ];
   const Index firstSegmentIdx = block.getFirstSegment();
   const Index begin = offsets[ firstSegmentIdx ];

   if( block.getType() == detail::Type::STREAM )  // Stream kernel - many short segments per warp
   {
      const Index warpIdx = threadIdx.x / Backend::getWarpSize();
      const Index end = begin + block.getSize();

      // Stream data to shared memory
      for( Index globalIdx = laneIdx + begin; globalIdx < end; globalIdx += WarpSize )
         streamShared[ warpIdx ][ globalIdx - begin ] = fetch( globalIdx );
      __syncwarp();
      const Index lastSegmentIdx = firstSegmentIdx + block.getSegmentsInBlock();

      for( Index i = firstSegmentIdx + laneIdx; i < lastSegmentIdx; i += WarpSize ) {
         const Index sharedEnd = offsets[ i + 1 ] - begin;  // end of preprocessed data
         result = identity;
         // Scalar reduction
         for( Index sharedIdx = offsets[ i ] - begin; sharedIdx < sharedEnd; sharedIdx++ )
            result = reduction( result, streamShared[ warpIdx ][ sharedIdx ] );
         keep( i, result );
      }
   }
   else if( block.getType() == detail::Type::VECTOR )  // Vector kernel - one segment per warp
   {
      const Index end = begin + block.getSize();
      const Index segmentIdx = block.getFirstSegment();

      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += WarpSize )
         result = reduction( result, fetch( globalIdx ) );

      // Parallel reduction
      using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduction, ReturnType >;
      result = BlockReduce::warpReduce( reduction, result );

      if( laneIdx == 0 )
         keep( segmentIdx, result );
   }
   else  // block.getType() == Type::LONG - several warps per segment
   {
      const Index segmentIdx = block.getFirstSegment();  // block.index[0];
      const Index end = offsets[ segmentIdx + 1 ];

      TNL_ASSERT_GT( block.getWarpsCount(), 0, "" );
      result = identity;
      for( Index globalIdx = begin + laneIdx + Backend::getWarpSize() * block.getWarpIdx(); globalIdx < end;
           globalIdx += Backend::getWarpSize() * block.getWarpsCount() )
      {
         result = reduction( result, fetch( globalIdx ) );
      }

      // Parallel reduction
      using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduction, ReturnType >;
      result = BlockReduce::warpReduce( reduction, result );

      const Index warpIdx = threadIdx.x / Backend::getWarpSize();
      if( laneIdx == 0 )
         multivectorShared[ warpIdx ] = result;

      __syncthreads();
      // Reduction in multivectorShared
      if( block.getWarpIdx() == 0 && laneIdx < 16 ) {
         constexpr int totalWarps = CudaBlockSize / WarpSize;
         if( totalWarps >= 32 ) {
            multivectorShared[ laneIdx ] = reduction( multivectorShared[ laneIdx ], multivectorShared[ laneIdx + 16 ] );
            __syncwarp();
         }
         if( totalWarps >= 16 ) {
            multivectorShared[ laneIdx ] = reduction( multivectorShared[ laneIdx ], multivectorShared[ laneIdx + 8 ] );
            __syncwarp();
         }
         if( totalWarps >= 8 ) {
            multivectorShared[ laneIdx ] = reduction( multivectorShared[ laneIdx ], multivectorShared[ laneIdx + 4 ] );
            __syncwarp();
         }
         if( totalWarps >= 4 ) {
            multivectorShared[ laneIdx ] = reduction( multivectorShared[ laneIdx ], multivectorShared[ laneIdx + 2 ] );
            __syncwarp();
         }
         if( totalWarps >= 2 ) {
            multivectorShared[ laneIdx ] = reduction( multivectorShared[ laneIdx ], multivectorShared[ laneIdx + 1 ] );
            __syncwarp();
         }
         if( laneIdx == 0 ) {
            // printf( "Long: segmentIdx %d -> %d \n", segmentIdx, multivectorShared[ 0 ] );
            keep( segmentIdx, multivectorShared[ 0 ] );
         }
      }
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail