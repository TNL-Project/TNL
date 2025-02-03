// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

// TODO: The following vector kernel is special case of the general variabel vector kernel.
// Check the performance and if it is the same, we can erase this kernel.
template< typename Segments, typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
__global__
void
reduceSegmentsCSRVectorKernel( Index gridIdx,
                               const Segments segments,
                               Index begin,
                               Index end,
                               Fetch fetch,
                               const Reduction reduction,
                               ResultKeeper keep,
                               const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   // We map one warp to each segment
   const Index segmentIdx = Backend::getGlobalThreadIdx_x( gridIdx ) / Backend::getWarpSize() + begin;
   if( segmentIdx >= end )
      return;

   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, segments.getOffsets().getSize(), "" );
   Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   ReturnType result = identity;
   for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + localIdx; globalIdx < endIdx;
        globalIdx += Backend::getWarpSize() )
   {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      result = reduction( result, detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ) );
      localIdx += Backend::getWarpSize();
   }

   // Reduction in each warp which means in each segment.
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduction, ReturnType >;
   result = BlockReduce::warpReduce( reduction, result );

   // Write the result
   if( laneIdx == 0 )
      keep( segmentIdx, result );
#endif
}

template< int ThreadsPerSegment,
          typename Segments,
          typename Index,
          typename Fetch,
          typename Reduce,
          typename Keep,
          typename Value >
__global__
void
reduceSegmentsCSRVariableVectorKernel( const Index gridID,
                                       const Segments segments,
                                       const Index begin,
                                       const Index end,
                                       Fetch fetch,
                                       Reduce reduce,
                                       Keep keep,
                                       const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index warpID =
      begin + ( ( gridID * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( warpID >= end )
      return;

   ReturnType result = identity;
   const Index laneID = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   Index endID = segments.getOffsets()[ warpID + 1 ];

   // Calculate result
   if constexpr( argumentCount< Fetch >() == 3 ) {
      Index localIdx = laneID;
      for( Index globalIdx = segments.getOffsets()[ warpID ] + laneID; globalIdx < endID; globalIdx += ThreadsPerSegment )
         result = reduce( result, fetch( warpID, localIdx, globalIdx ) );
      localIdx += ThreadsPerSegment;
   }
   else {
      for( Index globalIdx = segments.getOffsets()[ warpID ] + laneID; globalIdx < endID; globalIdx += ThreadsPerSegment )
         result = reduce( result, fetch( globalIdx ) );
   }

   // Parallel reduction
   #if defined( __HIP__ )
   if( ThreadsPerSegment > 16 ) {
      result = reduce( result, __shfl_down( result, 16 ) );
      result = reduce( result, __shfl_down( result, 8 ) );
      result = reduce( result, __shfl_down( result, 4 ) );
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 8 ) {
      result = reduce( result, __shfl_down( result, 8 ) );
      result = reduce( result, __shfl_down( result, 4 ) );
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 4 ) {
      result = reduce( result, __shfl_down( result, 4 ) );
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 2 ) {
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 1 )
      result = reduce( result, __shfl_down( result, 1 ) );
   #else
   if( ThreadsPerSegment > 16 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 8 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 4 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 2 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 1 )
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   #endif

   // Write the result
   if( laneID == 0 )
      keep( warpID, result );
#endif
}

template< int BlockSize,
          int ThreadsPerSegment,
          typename Segments,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
reduceSegmentsCSRLightMultivectorKernel( int gridIdx,
                                         const Segments segments,
                                         Index begin,
                                         Index end,
                                         Fetch fetch,
                                         const Reduction reduce,
                                         ResultKeeper keep,
                                         const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment + begin;
   if( segmentIdx >= end )
      return;

   __shared__ ReturnType shared[ BlockSize / Backend::getWarpSize() ];
   if( threadIdx.x < BlockSize / Backend::getWarpSize() )
      shared[ threadIdx.x ] = identity;  // (*) we sychronize threads later

   const int laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );             // & is cheaper than %
   const int inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   const Index beginIdx = segments.getOffsets()[ segmentIdx ];
   const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

   ReturnType result = identity;
   Index localIdx = laneIdx;
   for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
      result = reduce( result, detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ) );
      localIdx += ThreadsPerSegment;
   }

   #if defined( __HIP__ )
   result = reduce( result, __shfl_down( result, 16 ) );
   result = reduce( result, __shfl_down( result, 8 ) );
   result = reduce( result, __shfl_down( result, 4 ) );
   result = reduce( result, __shfl_down( result, 2 ) );
   result = reduce( result, __shfl_down( result, 1 ) );
   #else
   result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
   result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
   result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
   result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
   result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   #endif

   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   __syncthreads();  // Synchronize before writing to shared memory at (*)
   if( inWarpLaneIdx == 0 )
      shared[ warpIdx ] = result;

   __syncthreads();
   // Reduction in shared
   if( warpIdx == 0 && inWarpLaneIdx < 16 ) {
      // constexpr int totalWarps = BlockSize / Backend::getWarpSize();
      constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
      if( warpsPerSegment >= 32 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 16 ] );
         __syncwarp();
      }
      if( warpsPerSegment >= 16 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 8 ] );
         __syncwarp();
      }
      if( warpsPerSegment >= 8 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 4 ] );
         __syncwarp();
      }
      if( warpsPerSegment >= 4 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 2 ] );
         __syncwarp();
      }
      if( warpsPerSegment >= 2 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 1 ] );
         __syncwarp();
      }
      constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
      if( inWarpLaneIdx < segmentsCount && segmentIdx + inWarpLaneIdx < end ) {
         keep( segmentIdx + inWarpLaneIdx, shared[ inWarpLaneIdx * ThreadsPerSegment / Backend::getWarpSize() ] );
      }
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
