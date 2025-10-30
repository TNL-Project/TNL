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

   const Index segmentIdx =
      begin + ( ( gridID * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( segmentIdx >= end )
      return;

   ReturnType result = identity;
   const Index laneID = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   Index endID = segments.getOffsets()[ segmentIdx + 1 ];

   // Calculate result
   if constexpr( argumentCount< Fetch >() == 3 ) {
      Index localIdx = laneID;
      for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID; globalIdx += ThreadsPerSegment )
         result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
      localIdx += ThreadsPerSegment;
   }
   else {
      for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID; globalIdx += ThreadsPerSegment )
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
      keep( segmentIdx, result );
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
   constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
   constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();

   const Index segmentIdx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment + begin;
   if( segmentIdx >= end )
      return;

   __shared__ ReturnType shared[ BlockSize / Backend::getWarpSize() ];

   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );             // & is cheaper than %
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
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
   __syncthreads();
   if( inWarpLaneIdx == 0 )
      shared[ warpIdx ] = result;

   __syncthreads();
   // Reduction in shared memory
   if( warpIdx == 0 && inWarpLaneIdx < BlockSize / Backend::getWarpSize() ) {
      constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
      if constexpr( warpsPerSegment >= 32 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 16 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 16 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 8 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 8 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 4 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 4 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 2 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 2 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 1 ] );
         __syncwarp();
      }
      if( warpIdx == 0                      // first warp stores the results
          && inWarpLaneIdx < segmentsCount  // each thread in the warp handles one segment
          && segmentIdx + inWarpLaneIdx < end )
      {
         keep( segmentIdx + inWarpLaneIdx, shared[ inWarpLaneIdx * warpsPerSegment ] );
      }
   }
#endif
}

// Reduction with segment indexes

// TODO: The following vector kernel is special case of the general variabel vector kernel.
// Check the performance and if it is the same, we can erase this kernel.
template< typename Segments,
          typename ArrayView,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
reduceSegmentsCSRVectorKernelWithIndexes( Index gridIdx,
                                          const Segments segments,
                                          const ArrayView segmentIndexes,
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
   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridIdx ) / Backend::getWarpSize() + begin;
   if( segmentIdx_idx >= end )
      return;

   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
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
      keep( segmentIdx_idx, segmentIdx, result );

#endif
}

template< int ThreadsPerSegment,
          typename Segments,
          typename ArrayView,
          typename Index,
          typename Fetch,
          typename Reduce,
          typename Keep,
          typename Value >
__global__
void
reduceSegmentsCSRVariableVectorKernelWithIndexes( const Index gridID,
                                                  const Segments segments,
                                                  const ArrayView segmentIndexes,
                                                  const Index begin,
                                                  const Index end,
                                                  Fetch fetch,
                                                  Reduce reduce,
                                                  Keep keep,
                                                  const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx_idx =
      begin + ( ( gridID * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( segmentIdx_idx >= end )
      return;

   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   ReturnType result = identity;
   const Index laneID = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   Index endID = segments.getOffsets()[ segmentIdx + 1 ];

   // Calculate result
   if constexpr( argumentCount< Fetch >() == 3 ) {
      Index localIdx = laneID;
      for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID; globalIdx += ThreadsPerSegment )
         result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
      localIdx += ThreadsPerSegment;
   }
   else {
      for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID; globalIdx += ThreadsPerSegment )
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
      keep( segmentIdx_idx, segmentIdx, result );
#endif
}

template< int BlockSize,
          int ThreadsPerSegment,
          typename Segments,
          typename ArrayView,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
reduceSegmentsCSRLightMultivectorKernelWithIndexes( int gridIdx,
                                                    const Segments segments,
                                                    const ArrayView segmentIndexes,
                                                    Index begin,
                                                    Index end,
                                                    Fetch fetch,
                                                    const Reduction reduce,
                                                    ResultKeeper keep,
                                                    const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment + begin;
   if( segmentIdx_idx >= end )
      return;

   __shared__ ReturnType shared[ BlockSize / Backend::getWarpSize() ];
   if( threadIdx.x < BlockSize / Backend::getWarpSize() )
      shared[ threadIdx.x ] = identity;  // (*) we sychronize threads later

   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );             // & is cheaper than %
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
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
   // Reduction in shared memory
   if( warpIdx == 0 && inWarpLaneIdx < 16 ) {
      constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
      if constexpr( warpsPerSegment >= 32 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 16 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 16 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 8 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 8 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 4 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 4 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 2 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 2 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 1 ] );
         __syncwarp();
      }
      constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
      if( inWarpLaneIdx < segmentsCount && segmentIdx_idx + inWarpLaneIdx < end ) {
         keep( segmentIdx_idx,
               segmentIndexes[ segmentIdx_idx + inWarpLaneIdx ],
               shared[ inWarpLaneIdx * ThreadsPerSegment / Backend::getWarpSize() ] );
      }
   }
#endif
}

// Reduction with argument

// TODO: The following vector kernel is special case of the general variable vector kernel.
// Check the performance and if it is the same, we can erase this kernel.
template< typename Segments, typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
__global__
void
reduceSegmentsCSRVectorKernelWithArgument( Index gridIdx,
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
   Index argument = 0;
   ReturnType result = identity;
   for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + localIdx; globalIdx < endIdx;
        globalIdx += Backend::getWarpSize() )
   {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      reduction( result,
                 detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                 argument,
                 localIdx );
      localIdx += Backend::getWarpSize();
   }

   // Reduction in each warp which means in each segment.
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduction, result, argument );

   // Write the result
   if( laneIdx == 0 )
      keep( segmentIdx, argument_, result_ );
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
reduceSegmentsCSRVariableVectorKernelWithArgument( const Index gridID,
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

   const Index segmentIdx =
      begin + ( ( gridID * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( segmentIdx >= end )
      return;

   ReturnType result = identity;
   const Index laneID = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   Index endID = segments.getOffsets()[ segmentIdx + 1 ];

   // Calculate result
   Index localIdx = laneID;
   Index argument = 0;
   for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID; globalIdx += ThreadsPerSegment ) {
      reduce( result,
              detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
              argument,
              localIdx );
      localIdx += ThreadsPerSegment;
   }

   // Parallel reduction
   #if defined( __HIP__ )
   if( ThreadsPerSegment > 16 ) {
      reduce( result, __shfl_down( result, 16 ), argument, __shfl_down( argument, 16 ) );
      reduce( result, __shfl_down( result, 8 ), argument, __shfl_down( argument, 8 ) );
      reduce( result, __shfl_down( result, 4 ), argument, __shfl_down( argument, 4 ) );
      reduce( result, __shfl_down( result, 2 ), argument, __shfl_down( argument, 2 ) );
      reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   }
   else if( ThreadsPerSegment > 8 ) {
      reduce( result, __shfl_down( result, 8 ), argument, __shfl_down( argument, 8 ) );
      reduce( result, __shfl_down( result, 4 ), argument, __shfl_down( argument, 4 ) );
      reduce( result, __shfl_down( result, 2 ), argument, __shfl_down( argument, 2 ) );
      reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   }
   else if( ThreadsPerSegment > 4 ) {
      reduce( result, __shfl_down( result, 4 ), argument, __shfl_down( argument, 4 ) );
      reduce( result, __shfl_down( result, 2 ), argument, __shfl_down( argument, 2 ) );
      reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   }
   else if( ThreadsPerSegment > 2 ) {
      reduce( result, __shfl_down( result, 2 ), argument, __shfl_down( argument, 2 ) );
      reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   }
   else if( ThreadsPerSegment > 1 )
      reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   #else
   if( ThreadsPerSegment > 16 ) {
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 16 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 8 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 4 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 2 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );
   }
   else if( ThreadsPerSegment > 8 ) {
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 8 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 4 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 2 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );
   }
   else if( ThreadsPerSegment > 4 ) {
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 4 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 2 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );
   }
   else if( ThreadsPerSegment > 2 ) {
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 2 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );
   }
   else if( ThreadsPerSegment > 1 )
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );

   #endif

   // Write the result
   if( laneID == 0 )
      keep( segmentIdx, argument, result );
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
reduceSegmentsCSRLightMultivectorKernelWithArgument( int gridIdx,
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
   constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
   constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();

   const Index segmentIdx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment + begin;
   if( segmentIdx >= end )
      return;

   __shared__ ReturnType shared_results[ BlockSize / Backend::getWarpSize() ];
   __shared__ Index shared_arguments[ BlockSize / Backend::getWarpSize() ];

   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );             // & is cheaper than %
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   const Index beginIdx = segments.getOffsets()[ segmentIdx ];
   const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

   ReturnType result = identity;
   Index argument = 0;
   Index localIdx = laneIdx;
   for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
      reduce( result,
              detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
              argument,
              localIdx );
      localIdx += ThreadsPerSegment;
   }

   #if defined( __HIP__ )
   reduce( result, __shfl_down( result, 16 ), argument, __shfl_down( argument, 16 ) );
   reduce( result, __shfl_down( result, 8 ), argument, __shfl_down( argument, 8 ) );
   reduce( result, __shfl_down( result, 4 ), argument, __shfl_down( argument, 4 ) );
   reduce( result, __shfl_down( result, 2 ), argument, __shfl_down( argument, 2 ) );
   reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   #else
   reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 16 ) );
   reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 8 ) );
   reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 4 ) );
   reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 2 ) );
   reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );
   #endif

   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   __syncthreads();
   if( inWarpLaneIdx == 0 ) {
      shared_results[ warpIdx ] = result;
      shared_arguments[ warpIdx ] = argument;
   }

   __syncthreads();
   // Reduction in shared memory
   if( warpIdx == 0 && inWarpLaneIdx < BlockSize / Backend::getWarpSize() ) {
      if constexpr( warpsPerSegment >= 32 ) {
         reduce( shared_results[ inWarpLaneIdx ],
                 shared_results[ inWarpLaneIdx + 16 ],
                 shared_arguments[ inWarpLaneIdx ],
                 shared_arguments[ inWarpLaneIdx + 16 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 16 ) {
         reduce( shared_results[ inWarpLaneIdx ],
                 shared_results[ inWarpLaneIdx + 8 ],
                 shared_arguments[ inWarpLaneIdx ],
                 shared_arguments[ inWarpLaneIdx + 8 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 8 ) {
         reduce( shared_results[ inWarpLaneIdx ],
                 shared_results[ inWarpLaneIdx + 4 ],
                 shared_arguments[ inWarpLaneIdx ],
                 shared_arguments[ inWarpLaneIdx + 4 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 4 ) {
         reduce( shared_results[ inWarpLaneIdx ],
                 shared_results[ inWarpLaneIdx + 2 ],
                 shared_arguments[ inWarpLaneIdx ],
                 shared_arguments[ inWarpLaneIdx + 2 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 2 ) {
         reduce( shared_results[ inWarpLaneIdx ],
                 shared_results[ inWarpLaneIdx + 1 ],
                 shared_arguments[ inWarpLaneIdx ],
                 shared_arguments[ inWarpLaneIdx + 1 ] );
         __syncwarp();
      }

      __syncthreads();
      if( warpIdx == 0                      // first warp stores the results
          && inWarpLaneIdx < segmentsCount  // each thread in the warp handles one segment
          && segmentIdx + inWarpLaneIdx < end )
      {
         keep( segmentIdx + inWarpLaneIdx,
               shared_arguments[ inWarpLaneIdx * warpsPerSegment ],
               shared_results[ inWarpLaneIdx * warpsPerSegment ] );
      }
   }
#endif
}

// Reduction with segment indexes and argument

// TODO: The following vector kernel is special case of the general variabel vector kernel.
// Check the performance and if it is the same, we can erase this kernel.
template< typename Segments,
          typename ArrayView,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
reduceSegmentsCSRVectorKernelWithIndexesAndArgument( Index gridIdx,
                                                     const Segments segments,
                                                     const ArrayView segmentIndexes,
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
   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridIdx ) / Backend::getWarpSize() + begin;
   if( segmentIdx_idx >= end )
      return;

   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   TNL_ASSERT_LT( segmentIdx + 1, segments.getOffsets().getSize(), "" );
   Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   ReturnType result = identity;
   Index argument = 0;
   for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + localIdx; globalIdx < endIdx;
        globalIdx += Backend::getWarpSize() )
   {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      reduction( result,
                 detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                 argument,
                 localIdx );
      localIdx += Backend::getWarpSize();
   }
   // Reduction in each warp which means in each segment.
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduction, result, argument );

   // Write the result
   if( laneIdx == 0 )
      keep( segmentIdx_idx, segmentIdx, argument_, result_ );

#endif
}

template< int ThreadsPerSegment,
          typename Segments,
          typename ArrayView,
          typename Index,
          typename Fetch,
          typename Reduce,
          typename Keep,
          typename Value >
__global__
void
reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument( const Index gridID,
                                                             const Segments segments,
                                                             const ArrayView segmentIndexes,
                                                             const Index begin,
                                                             const Index end,
                                                             Fetch fetch,
                                                             Reduce reduce,
                                                             Keep keep,
                                                             const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx_idx =
      begin + ( ( gridID * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( segmentIdx_idx >= end )
      return;

   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   ReturnType result = identity;
   const Index laneID = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   Index endID = segments.getOffsets()[ segmentIdx + 1 ];

   // Calculate result
   Index localIdx = laneID;
   Index argument = 0;
   for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID; globalIdx += ThreadsPerSegment ) {
      reduce( result,
              detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
              argument,
              localIdx );
      localIdx += ThreadsPerSegment;
   }

   // Parallel reduction
   #if defined( __HIP__ )
   if( ThreadsPerSegment > 16 ) {
      reduce( result, __shfl_down( result, 16 ), argument, __shfl_down( argument, 16 ) );
      reduce( result, __shfl_down( result, 8 ), argument, __shfl_down( argument, 8 ) );
      reduce( result, __shfl_down( result, 4 ), argument, __shfl_down( argument, 4 ) );
      reduce( result, __shfl_down( result, 2 ), argument, __shfl_down( argument, 2 ) );
      reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   }
   else if( ThreadsPerSegment > 8 ) {
      reduce( result, __shfl_down( result, 8 ), argument, __shfl_down( argument, 8 ) );
      reduce( result, __shfl_down( result, 4 ), argument, __shfl_down( argument, 4 ) );
      reduce( result, __shfl_down( result, 2 ), argument, __shfl_down( argument, 2 ) );
      reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   }
   else if( ThreadsPerSegment > 4 ) {
      reduce( result, __shfl_down( result, 4 ), argument, __shfl_down( argument, 4 ) );
      reduce( result, __shfl_down( result, 2 ), argument, __shfl_down( argument, 2 ) );
      reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   }
   else if( ThreadsPerSegment > 2 ) {
      reduce( result, __shfl_down( result, 2 ), argument, __shfl_down( argument, 2 ) );
      reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   }
   else if( ThreadsPerSegment > 1 )
      reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   #else
   if( ThreadsPerSegment > 16 ) {
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 16 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 8 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 4 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 2 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );
   }
   else if( ThreadsPerSegment > 8 ) {
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 8 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 4 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 2 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );
   }
   else if( ThreadsPerSegment > 4 ) {
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 4 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 2 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );
   }
   else if( ThreadsPerSegment > 2 ) {
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 2 ) );
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );
   }
   else if( ThreadsPerSegment > 1 )
      reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );

   #endif

   // Write the result
   if( laneID == 0 )
      keep( segmentIdx_idx, segmentIdx, argument, result );
#endif
}

template< int BlockSize,
          int ThreadsPerSegment,
          typename Segments,
          typename ArrayView,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
reduceSegmentsCSRLightMultivectorKernelWithIndexesAndArgument( int gridIdx,
                                                               const Segments segments,
                                                               const ArrayView segmentIndexes,
                                                               Index begin,
                                                               Index end,
                                                               Fetch fetch,
                                                               const Reduction reduce,
                                                               ResultKeeper keep,
                                                               const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
   constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();

   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment + begin;
   if( segmentIdx_idx >= end )
      return;

   __shared__ ReturnType shared_results[ BlockSize / Backend::getWarpSize() ];
   __shared__ Index shared_arguments[ BlockSize / Backend::getWarpSize() ];

   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );             // & is cheaper than %
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   const Index beginIdx = segments.getOffsets()[ segmentIdx ];
   const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

   ReturnType result = identity;
   Index argument = 0;
   Index localIdx = laneIdx;
   for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
      reduce( result,
              detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
              argument,
              localIdx );
      localIdx += ThreadsPerSegment;
   }

   #if defined( __HIP__ )
   reduce( result, __shfl_down( result, 16 ), argument, __shfl_down( argument, 16 ) );
   reduce( result, __shfl_down( result, 8 ), argument, __shfl_down( argument, 8 ) );
   reduce( result, __shfl_down( result, 4 ), argument, __shfl_down( argument, 4 ) );
   reduce( result, __shfl_down( result, 2 ), argument, __shfl_down( argument, 2 ) );
   reduce( result, __shfl_down( result, 1 ), argument, __shfl_down( argument, 1 ) );
   #else
   reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 16 ) );
   reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 8 ) );
   reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 4 ) );
   reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 2 ) );
   reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ), argument, __shfl_down_sync( 0xFFFFFFFF, argument, 1 ) );
   #endif

   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   __syncthreads();
   if( inWarpLaneIdx == 0 ) {
      shared_results[ warpIdx ] = result;
      shared_arguments[ warpIdx ] = argument;
   }

   __syncthreads();
   // Reduction in shared memory
   if( warpIdx == 0 && inWarpLaneIdx < BlockSize / Backend::getWarpSize() ) {
      if constexpr( warpsPerSegment >= 32 ) {
         reduce( shared_results[ inWarpLaneIdx ],
                 shared_results[ inWarpLaneIdx + 16 ],
                 shared_arguments[ inWarpLaneIdx ],
                 shared_arguments[ inWarpLaneIdx + 16 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 16 ) {
         reduce( shared_results[ inWarpLaneIdx ],
                 shared_results[ inWarpLaneIdx + 8 ],
                 shared_arguments[ inWarpLaneIdx ],
                 shared_arguments[ inWarpLaneIdx + 8 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 8 ) {
         reduce( shared_results[ inWarpLaneIdx ],
                 shared_results[ inWarpLaneIdx + 4 ],
                 shared_arguments[ inWarpLaneIdx ],
                 shared_arguments[ inWarpLaneIdx + 4 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 4 ) {
         reduce( shared_results[ inWarpLaneIdx ],
                 shared_results[ inWarpLaneIdx + 2 ],
                 shared_arguments[ inWarpLaneIdx ],
                 shared_arguments[ inWarpLaneIdx + 2 ] );
         __syncwarp();
      }
      if constexpr( warpsPerSegment >= 2 ) {
         reduce( shared_results[ inWarpLaneIdx ],
                 shared_results[ inWarpLaneIdx + 1 ],
                 shared_arguments[ inWarpLaneIdx ],
                 shared_arguments[ inWarpLaneIdx + 1 ] );
         __syncwarp();
      }
      if( warpIdx == 0                      // first warp stores the results
          && inWarpLaneIdx < segmentsCount  // each thread in the warp handles one segment
          && segmentIdx_idx + inWarpLaneIdx < end )
      {
         keep( segmentIdx_idx,
               segmentIndexes[ segmentIdx_idx + inWarpLaneIdx ],
               shared_arguments[ inWarpLaneIdx * warpsPerSegment ],
               shared_results[ inWarpLaneIdx * warpsPerSegment ] );
      }
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
