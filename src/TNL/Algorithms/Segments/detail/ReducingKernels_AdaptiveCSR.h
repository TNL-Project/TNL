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
          typename ResultStorer,
          typename Value >
__global__
void
reduceSegmentsCSRAdaptiveKernel( int gridIdx,
                                 BlocksView blocks,
                                 Offsets offsets,
                                 Fetch fetch,
                                 Reduction reduction,
                                 ResultStorer store,
                                 Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   constexpr int CudaBlockSize = detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::CudaBlockSize();
   constexpr int WarpSize = Backend::getWarpSize();
   constexpr int WarpsCount = detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::WarpsCount();
   constexpr std::size_t StreamedSharedElementsPerWarp =
      detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::StreamedSharedElementsPerWarp();

   __shared__ ReturnType streamShared[ WarpsCount ][ StreamedSharedElementsPerWarp ];
   __shared__ ReturnType multivectorShared[ CudaBlockSize / WarpSize ];

   const Index index = ( ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x ) + threadIdx.x;
   const Index blockIdx = index / WarpSize;
   if( blockIdx >= blocks.getSize() - 1 )
      return;

   if( threadIdx.x < CudaBlockSize / WarpSize )
      multivectorShared[ threadIdx.x ] = identity;
   __syncthreads();
   ReturnType result = identity;
   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
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
         store( i, result );
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
         store( segmentIdx, result );
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
            //printf( "Long: segmentIdx %d -> %d \n", segmentIdx, multivectorShared[ 0 ] );
            store( segmentIdx, multivectorShared[ 0 ] );
         }
      }
   }
#endif
}

template< typename BlocksView,
          typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value >
__global__
void
reduceSegmentsCSRAdaptiveKernelWithArgument( int gridIdx,
                                             BlocksView blocks,
                                             Offsets offsets,
                                             Fetch fetch,
                                             Reduction reduction,
                                             ResultStorer store,
                                             Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   constexpr int CudaBlockSize = detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::CudaBlockSize();
   constexpr int WarpSize = Backend::getWarpSize();
   constexpr int WarpsCount = detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::WarpsCount();
   constexpr std::size_t StreamedSharedElementsPerWarp =
      detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::StreamedSharedElementsPerWarp();

   __shared__ ReturnType streamShared_result[ WarpsCount ][ StreamedSharedElementsPerWarp ];
   __shared__ ReturnType multivectorShared_result[ CudaBlockSize / WarpSize ];
   __shared__ Index multivectorShared_argument[ CudaBlockSize / WarpSize ];

   const Index index = ( ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x ) + threadIdx.x;
   const Index blockIdx = index / WarpSize;
   if( blockIdx >= blocks.getSize() - 1 )
      return;

   if( threadIdx.x < CudaBlockSize / WarpSize )
      multivectorShared_result[ threadIdx.x ] = identity;
   __syncthreads();
   ReturnType result = identity;
   Index argument = 0;
   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   const auto& block = blocks[ blockIdx ];
   const Index firstSegmentIdx = block.getFirstSegment();
   const Index begin = offsets[ firstSegmentIdx ];

   if( block.getType() == detail::Type::STREAM )  // Stream kernel - many short segments per warp
   {
      const Index warpIdx = threadIdx.x / Backend::getWarpSize();
      const Index end = begin + block.getSize();

      // Stream data to shared memory
      for( Index globalIdx = laneIdx + begin; globalIdx < end; globalIdx += WarpSize )
         streamShared_result[ warpIdx ][ globalIdx - begin ] = fetch( globalIdx );

      __syncwarp();
      const Index lastSegmentIdx = firstSegmentIdx + block.getSegmentsInBlock();

      for( Index i = firstSegmentIdx + laneIdx; i < lastSegmentIdx; i += WarpSize ) {
         const Index sharedEnd = offsets[ i + 1 ] - begin;  // end of preprocessed data
         result = identity;
         // Scalar reduction
         Index localIdx = 0;
         for( Index sharedIdx = offsets[ i ] - begin; sharedIdx < sharedEnd; sharedIdx++, localIdx++ )
            reduction( result, streamShared_result[ warpIdx ][ sharedIdx ], argument, localIdx );
         store( i, argument, result );
      }
   }
   else if( block.getType() == detail::Type::VECTOR )  // Vector kernel - one segment per warp
   {
      const Index end = begin + block.getSize();
      const Index segmentIdx = block.getFirstSegment();

      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += WarpSize )
         reduction( result, fetch( globalIdx ), argument, globalIdx - begin );

      // Parallel reduction
      using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
      auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduction, result, argument );

      if( laneIdx == 0 )
         store( segmentIdx, argument_, result_ );
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
         reduction( result, fetch( globalIdx ), argument, globalIdx - begin );
      }

      // Parallel reduction
      using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
      auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduction, result, argument );

      const Index warpIdx = threadIdx.x / Backend::getWarpSize();
      if( laneIdx == 0 ) {
         multivectorShared_result[ warpIdx ] = result_;
         multivectorShared_argument[ warpIdx ] = argument_;
      }

      __syncthreads();
      // Reduction in multivectorShared
      if( block.getWarpIdx() == 0 && laneIdx < 16 ) {
         constexpr int totalWarps = CudaBlockSize / WarpSize;
         if( totalWarps >= 32 ) {
            reduction( multivectorShared_result[ laneIdx ],
                       multivectorShared_result[ laneIdx + 16 ],
                       multivectorShared_argument[ laneIdx ],
                       multivectorShared_argument[ laneIdx + 16 ] );
            __syncwarp();
         }
         if( totalWarps >= 16 ) {
            reduction( multivectorShared_result[ laneIdx ],
                       multivectorShared_result[ laneIdx + 8 ],
                       multivectorShared_argument[ laneIdx ],
                       multivectorShared_argument[ laneIdx + 8 ] );
            __syncwarp();
         }
         if( totalWarps >= 8 ) {
            reduction( multivectorShared_result[ laneIdx ],
                       multivectorShared_result[ laneIdx + 4 ],
                       multivectorShared_argument[ laneIdx ],
                       multivectorShared_argument[ laneIdx + 4 ] );
            __syncwarp();
         }
         if( totalWarps >= 4 ) {
            reduction( multivectorShared_result[ laneIdx ],
                       multivectorShared_result[ laneIdx + 2 ],
                       multivectorShared_argument[ laneIdx ],
                       multivectorShared_argument[ laneIdx + 2 ] );
            __syncwarp();
         }
         if( totalWarps >= 2 ) {
            reduction( multivectorShared_result[ laneIdx ],
                       multivectorShared_result[ laneIdx + 1 ],
                       multivectorShared_argument[ laneIdx ],
                       multivectorShared_argument[ laneIdx + 1 ] );
            __syncwarp();
         }
         if( laneIdx == 0 ) {
            //printf( "Long: segmentIdx %d -> %d \n", segmentIdx, multivectorShared_result[ 0 ] );
            store( segmentIdx, multivectorShared_argument[ 0 ], multivectorShared_result[ 0 ] );
         }
      }
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
