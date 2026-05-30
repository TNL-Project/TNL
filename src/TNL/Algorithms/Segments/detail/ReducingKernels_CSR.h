// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/detail/CudaReductionKernel.h>
#include <TNL/Algorithms/Segments/detail/FetchLambdaAdapter.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

// TODO: The following vector kernel is special case of the general variable vector kernel.
// Check the performance and if it is the same, we can erase this kernel.
template< typename Segments, typename Index, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
__global__
void
reduceSegmentsCSRVectorKernel(
   Index gridIdx,
   const Segments segments,
   Index begin,
   Index end,
   Fetch fetch,
   const Reduction reduction,
   ResultStorer store,
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
      store( segmentIdx, result );
#endif
}

template<
   int ThreadsPerSegment,
   typename Segments,
   typename Index,
   typename Fetch,
   typename Reduce,
   typename Store,
   typename Value >
__global__
void
reduceSegmentsCSRVariableVectorKernel(
   const Index gridID,
   const Segments segments,
   const Index begin,
   const Index end,
   Fetch fetch,
   Reduce reduce,
   Store store,
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
   if constexpr( callableArgumentCount< Fetch >() == 3 ) {
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
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduce, ReturnType >;
   result = BlockReduce::template warpReduce< ThreadsPerSegment >( reduce, result );

   // Write the result
   if( laneID == 0 )
      store( segmentIdx, result );
#endif
}

template<
   int BlockSize,
   int ThreadsPerSegment,
   typename Segments,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegmentsCSRLightMultivectorKernel(
   int gridIdx,
   const Segments segments,
   Index begin,
   Index end,
   Fetch fetch,
   const Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment + begin;
   if( segmentIdx >= end )
      return;

   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   const Index beginIdx = segments.getOffsets()[ segmentIdx ];
   const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

   ReturnType result = identity;
   Index localIdx = laneIdx;
   for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
      result = reduce( result, detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ) );
      localIdx += ThreadsPerSegment;
   }

   // Parallel reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< BlockSize, Reduction, ReturnType >;
   result = BlockReduce::warpReduce( reduce, result );

   constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
   constexpr int warpsCount = BlockSize / Backend::getWarpSize();
   constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   __shared__ ReturnType shared[ warpsCount ];

   // Write results of parallel reduction to shared memory
   __syncthreads();
   if( inWarpLaneIdx == 0 )
      shared[ warpIdx ] = result;

   // The first warp performs the remaining reduction
   __syncthreads();
   if( warpIdx == 0 ) {
      ReturnType partial = inWarpLaneIdx < warpsCount ? shared[ inWarpLaneIdx ] : identity;
      partial = BlockReduce::template warpReduce< warpsPerSegment >( reduce, partial );
      // Only the first thread in each group has the correct result
      const int groupIdx = inWarpLaneIdx / warpsPerSegment;
      if( inWarpLaneIdx % warpsPerSegment == 0 && groupIdx < segmentsCount && segmentIdx + groupIdx < end )
         store( segmentIdx + groupIdx, partial );
   }
#endif
}

template<
   typename Segments,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value,
   int BlockSize = 256 >
__global__
void
reduceSegmentsCSRDynamicGroupingKernel(
   int gridIdx,
   const Index threadsPerSegment,
   const Segments segments,
   Index begin,
   Index end,
   Fetch fetch,
   const Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   constexpr Index warpSize = Backend::getWarpSize();
   constexpr Index warpsPerBlock = BlockSize / warpSize;
   constexpr Index none_scheduled = std::numeric_limits< Index >::max();
   __shared__ Index warps_scheduler[ BlockSize ];
   const auto& offsets = segments.getOffsets();

   const Index segmentIdx =
      threadIdx.x < ( BlockSize / threadsPerSegment )
         ? begin + ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * ( BlockSize / threadsPerSegment ) + threadIdx.x
         : none_scheduled;
   bool reduce_segment = ( segmentIdx < end && threadIdx.x < BlockSize / threadsPerSegment );

   // Processing segments larger than BlockSize
   __shared__ Index scheduled_segment[ 1 ];

   Index segment_size = -1;
   if( reduce_segment ) {
      segment_size = offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
   }

   if( threadIdx.x == 0 )
      *scheduled_segment = none_scheduled;
   __syncthreads();
   while( true ) {
      if( reduce_segment && segment_size > BlockSize ) {
         AtomicOperations< Devices::GPU >::CAS( *scheduled_segment, *scheduled_segment, segmentIdx );
      }
      __syncthreads();
      if( *scheduled_segment == none_scheduled )
         break;

      ReturnType result = identity;
      Index globalIdx = offsets[ *scheduled_segment ] + threadIdx.x;
      const Index endIdx = offsets[ *scheduled_segment + 1 ];
      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         Index localIdx = threadIdx.x;
         while( globalIdx < endIdx ) {
            result = reduce( result, fetch( *scheduled_segment, localIdx, globalIdx ) );
            localIdx += BlockSize;
            globalIdx += BlockSize;
         }
      }
      else
         while( globalIdx < endIdx ) {
            result = reduce( result, fetch( globalIdx ) );
            globalIdx += BlockSize;
         }

      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< BlockSize, Reduction, ReturnType >;
      __shared__ typename BlockReduce::Storage storage;

      result = BlockReduce::reduce( reduce, identity, result, storage, threadIdx.x );

      // Write the result
      if( threadIdx.x == 0 ) {
         TNL_ASSERT_NE( *scheduled_segment, none_scheduled, "" );
         store( *scheduled_segment, result );
      }
      __syncthreads();

      // Mark segment as processed
      if( segmentIdx == *scheduled_segment ) {
         reduce_segment = false;
         *scheduled_segment = none_scheduled;
      }
      __syncthreads();
   }

   // Processing segments smaller than BlockSize and larger the warp size
   __shared__ int active_warps[ 1 ];
   if( threadIdx.x == 0 )
      active_warps[ 0 ] = 0;
   __syncthreads();

   // Each thread owning segment with size larger than warpSize registers for scheduling
   if( reduce_segment && segment_size > warpSize ) {
      warps_scheduler[ AtomicOperations< Devices::GPU >::add( active_warps[ 0 ], 1 ) ] = segmentIdx;
      reduce_segment = false;
   }
   __syncthreads();

   // Now reduce scheduled segments in warps
   Index warp_idx = threadIdx.x / warpSize;

   while( warp_idx < active_warps[ 0 ] ) {
      Index scheduled_segment = warps_scheduler[ warp_idx ];
      Index globalIdx = offsets[ scheduled_segment ] + ( threadIdx.x & ( warpSize - 1 ) );  // & is cheaper than %
      const Index endIdx = offsets[ scheduled_segment + 1 ];
      ReturnType result = identity;
      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         Index localIdx = threadIdx.x & ( warpSize - 1 );  // & is cheaper than %
         for( ; globalIdx < endIdx; globalIdx += warpSize ) {
            result = reduce( result, fetch( scheduled_segment, localIdx, globalIdx ) );
            localIdx += warpSize;
         }
      }
      else {
         for( ; globalIdx < endIdx; globalIdx += warpSize ) {
            result = reduce( result, fetch( globalIdx ) );
         }
      }
      __syncthreads();
      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< BlockSize, Reduction, ReturnType >;
      result = BlockReduce::warpReduce( reduce, result );

      // Write the result
      if( ( threadIdx.x & ( warpSize - 1 ) ) == 0 ) {  // first lane in the warp
         TNL_ASSERT_NE( scheduled_segment, none_scheduled, "" );
         store( scheduled_segment, result );
      }
      warp_idx += warpsPerBlock;
   }

   // Processing segments smaller than or equal to warp size
   if( reduce_segment ) {
      Index globalIdx = offsets[ segmentIdx ];
      const Index endIdx = offsets[ segmentIdx + 1 ];
      ReturnType result = identity;
      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         Index localIdx = 0;
         for( ; globalIdx < endIdx; globalIdx++ ) {
            result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
            localIdx++;
         }
      }
      else {
         for( ; globalIdx < endIdx; globalIdx++ ) {
            result = reduce( result, fetch( globalIdx ) );
         }
      }
      // Write the result
      TNL_ASSERT_NE( segmentIdx, none_scheduled, "" );
      store( segmentIdx, result );
   }
#endif
}

// Reduction with segment indexes

// TODO: The following vector kernel is special case of the general variable vector kernel.
// Check the performance and if it is the same, we can erase this kernel.
template<
   typename Segments,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegmentsCSRVectorKernelWithIndexes(
   Index gridIdx,
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   const Reduction reduction,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   // We map one warp to each segment
   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridIdx ) / Backend::getWarpSize();
   if( segmentIdx_idx >= segmentIndexes.getSize() )
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
      store( segmentIdx_idx, segmentIdx, result );

#endif
}

template<
   int ThreadsPerSegment,
   typename Segments,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduce,
   typename Store,
   typename Value >
__global__
void
reduceSegmentsCSRVariableVectorKernelWithIndexes(
   const Index gridID,
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   Reduce reduce,
   Store store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx_idx =
      ( ( gridID * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( segmentIdx_idx >= segmentIndexes.getSize() )
      return;

   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   ReturnType result = identity;
   const Index laneID = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   Index endID = segments.getOffsets()[ segmentIdx + 1 ];

   // Calculate result
   if constexpr( callableArgumentCount< Fetch >() == 3 ) {
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
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduce, ReturnType >;
   result = BlockReduce::template warpReduce< ThreadsPerSegment >( reduce, result );

   // Write the result
   if( laneID == 0 )
      store( segmentIdx_idx, segmentIdx, result );
#endif
}

template<
   int BlockSize,
   int ThreadsPerSegment,
   typename Segments,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegmentsCSRLightMultivectorKernelWithIndexes(
   int gridIdx,
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   const Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment;
   if( segmentIdx_idx >= segmentIndexes.getSize() )
      return;

   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   const Index beginIdx = segments.getOffsets()[ segmentIdx ];
   const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

   ReturnType result = identity;
   Index localIdx = laneIdx;
   for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
      result = reduce( result, detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ) );
      localIdx += ThreadsPerSegment;
   }

   // Parallel reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< BlockSize, Reduction, ReturnType >;
   result = BlockReduce::warpReduce( reduce, result );

   constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
   constexpr int warpsCount = BlockSize / Backend::getWarpSize();
   constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   __shared__ ReturnType shared[ warpsCount ];

   // Write results of parallel reduction to shared memory
   __syncthreads();
   if( inWarpLaneIdx == 0 )
      shared[ warpIdx ] = result;

   // The first warp performs the remaining reduction
   __syncthreads();
   if( warpIdx == 0 ) {
      ReturnType partial = inWarpLaneIdx < warpsCount ? shared[ inWarpLaneIdx ] : identity;
      partial = BlockReduce::template warpReduce< warpsPerSegment >( reduce, partial );
      // Only the first thread in each group has the correct result
      const int groupIdx = inWarpLaneIdx / warpsPerSegment;
      if( inWarpLaneIdx % warpsPerSegment == 0 && groupIdx < segmentsCount
          && segmentIdx_idx + groupIdx < segmentIndexes.getSize() )
      {
         store( segmentIdx_idx, segmentIndexes[ segmentIdx_idx + groupIdx ], partial );
      }
   }
#endif
}

template<
   typename Segments,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value,
   int BlockSize = 256 >
__global__
void
reduceSegmentsCSRDynamicGroupingKernelWithIndexes(
   int gridIdx,
   const Index threadsPerSegment,
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   const Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   constexpr Index warpSize = Backend::getWarpSize();
   constexpr Index warpsPerBlock = BlockSize / warpSize;
   constexpr Index none_scheduled = std::numeric_limits< Index >::max();
   __shared__ Index warps_scheduler[ BlockSize ];
   const auto& offsets = segments.getOffsets();

   const Index segmentIdx_idx =
      threadIdx.x < ( BlockSize / threadsPerSegment )
         ? ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * ( BlockSize / threadsPerSegment ) + threadIdx.x
         : none_scheduled;
   bool reduce_segment = ( segmentIdx_idx < segmentIndexes.getSize() && threadIdx.x < BlockSize / threadsPerSegment );

   // Processing segments larger than BlockSize
   __shared__ Index scheduled_segment_idx[ 1 ];
   ReturnType result = identity;

   Index segment_size = -1;
   Index segmentIdx = -1;
   if( reduce_segment ) {
      segmentIdx = segmentIndexes[ segmentIdx_idx ];
      segment_size = offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
   }

   if( threadIdx.x == 0 )
      scheduled_segment_idx[ 0 ] = none_scheduled;
   __syncthreads();
   while( true ) {
      if( reduce_segment && segment_size > BlockSize ) {
         AtomicOperations< Devices::GPU >::CAS( scheduled_segment_idx[ 0 ], scheduled_segment_idx[ 0 ], segmentIdx_idx );
      }
      __syncthreads();
      if( scheduled_segment_idx[ 0 ] == none_scheduled )
         break;

      TNL_ASSERT_LT( scheduled_segment_idx[ 0 ], segmentIndexes.getSize(), "" );
      Index scheduled_segment = segmentIndexes[ scheduled_segment_idx[ 0 ] ];

      Index globalIdx = offsets[ scheduled_segment ];
      const Index endIdx = offsets[ scheduled_segment + 1 ];

      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         Index localIdx = threadIdx.x;
         while( globalIdx < endIdx ) {
            result = reduce( result, fetch( scheduled_segment, localIdx, globalIdx ) );
            localIdx += BlockSize;
            globalIdx += BlockSize;
         }
      }
      else
         while( globalIdx < endIdx ) {
            result = reduce( result, fetch( globalIdx ) );
            globalIdx += BlockSize;
         }
      if( segmentIdx_idx == scheduled_segment_idx[ 0 ] ) {
         reduce_segment = false;
         scheduled_segment_idx[ 0 ] = none_scheduled;
      }
      __syncthreads();

      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< BlockSize, Reduction, ReturnType >;
      __shared__ typename BlockReduce::Storage storage;

      result = BlockReduce::reduce( reduce, identity, result, storage, threadIdx.x );

      // Write the result
      if( threadIdx.x == 0 )
         store( scheduled_segment_idx[ 0 ], scheduled_segment, result );
   }

   // Processing segments smaller than BlockSize and larger the warp size
   __shared__ int active_warps[ 1 ];
   if( threadIdx.x == 0 )
      active_warps[ 0 ] = 0;
   __syncthreads();

   // Each thread owning segment with size larger than warpSize registers for scheduling
   if( reduce_segment && segment_size > warpSize ) {
      warps_scheduler[ AtomicOperations< Devices::GPU >::add( active_warps[ 0 ], 1 ) ] = segmentIdx_idx;
      reduce_segment = false;
   }
   __syncthreads();

   // Now reduce scheduled segments in warps
   Index warp_idx = threadIdx.x / warpSize;

   while( warp_idx < active_warps[ 0 ] ) {
      TNL_ASSERT_LT( warps_scheduler[ warp_idx ], segmentIndexes.getSize(), "" );
      Index scheduled_segment = segmentIndexes[ warps_scheduler[ warp_idx ] ];
      Index globalIdx = offsets[ scheduled_segment ] + ( threadIdx.x & ( warpSize - 1 ) );  // & is cheaper than %
      const Index endIdx = offsets[ scheduled_segment + 1 ];
      result = identity;
      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         Index localIdx = threadIdx.x & ( warpSize - 1 );  // & is cheaper than %
         for( ; globalIdx < endIdx; globalIdx += warpSize ) {
            result = reduce( result, fetch( scheduled_segment, localIdx, globalIdx ) );
            localIdx += warpSize;
         }
      }
      else {
         for( ; globalIdx < endIdx; globalIdx += warpSize ) {
            result = reduce( result, fetch( globalIdx ) );
         }
      }
      __syncthreads();
      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< BlockSize, Reduction, ReturnType >;
      result = BlockReduce::warpReduce( reduce, result );

      // Write the result
      if( ( threadIdx.x & ( warpSize - 1 ) ) == 0 )  // first lane in the warp
         store(
            warps_scheduler[ warp_idx ],  // segmentIdx_idx
            scheduled_segment,
            result );
      warp_idx += warpsPerBlock;
   }

   // Processing segments smaller than or equal to warp size
   if( reduce_segment ) {
      Index globalIdx = offsets[ segmentIdx ];
      const Index endIdx = offsets[ segmentIdx + 1 ];
      result = identity;
      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         Index localIdx = 0;
         for( ; globalIdx < endIdx; globalIdx++ ) {
            result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
            localIdx++;
         }
      }
      else {
         for( ; globalIdx < endIdx; globalIdx++ ) {
            result = reduce( result, fetch( globalIdx ) );
         }
      }
      // Write the result
      store( segmentIdx_idx, segmentIdx, result );
   }
#endif
}

// Reduction with argument

// TODO: The following vector kernel is special case of the general variable vector kernel.
// Check the performance and if it is the same, we can erase this kernel.
template< typename Segments, typename Index, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
__global__
void
reduceSegmentsCSRVectorKernelWithArgument(
   Index gridIdx,
   const Segments segments,
   Index begin,
   Index end,
   Fetch fetch,
   const Reduction reduction,
   ResultStorer store,
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
      reduction(
         result,
         detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
         argument,
         localIdx );
      localIdx += Backend::getWarpSize();
   }

   // Reduction in each warp which means in each segment.
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduction, result, argument );

   // Write the result
   if( laneIdx == 0 ) {
      bool emptySegment = ( segments.getOffsets()[ segmentIdx ] == endIdx );
      store( segmentIdx, argument_, result_, emptySegment );
   }
#endif
}

template<
   int ThreadsPerSegment,
   typename Segments,
   typename Index,
   typename Fetch,
   typename Reduce,
   typename Store,
   typename Value >
__global__
void
reduceSegmentsCSRVariableVectorKernelWithArgument(
   const Index gridID,
   const Segments segments,
   const Index begin,
   const Index end,
   Fetch fetch,
   Reduce reduce,
   Store store,
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
      reduce(
         result,
         detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
         argument,
         localIdx );
      localIdx += ThreadsPerSegment;
   }

   // Parallel reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduce, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::template warpReduceWithArgument< ThreadsPerSegment >( reduce, result, argument );

   // Write the result
   if( laneID == 0 ) {
      TNL_ASSERT_LT( segmentIdx + 1, segments.getOffsets().getSize(), "" );
      bool emptySegment = ( segments.getOffsets()[ segmentIdx ] == segments.getOffsets()[ segmentIdx + 1 ] );
      store( segmentIdx, argument_, result_, emptySegment );
   }
#endif
}

template<
   int BlockSize,
   int ThreadsPerSegment,
   typename Segments,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegmentsCSRLightMultivectorKernelWithArgument(
   int gridIdx,
   const Segments segments,
   Index begin,
   Index end,
   Fetch fetch,
   const Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment + begin;
   if( segmentIdx >= end )
      return;

   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   const Index beginIdx = segments.getOffsets()[ segmentIdx ];
   const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

   ReturnType result = identity;
   Index argument = 0;
   Index localIdx = laneIdx;
   for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
      reduce(
         result,
         detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
         argument,
         localIdx );
      localIdx += ThreadsPerSegment;
   }

   // Parallel reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< BlockSize, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduce, result, argument );

   constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
   constexpr int warpsCount = BlockSize / Backend::getWarpSize();
   constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   __shared__ ReturnType shared_results[ warpsCount ];
   __shared__ Index shared_arguments[ warpsCount ];

   // Write results of parallel reduction to shared memory
   __syncthreads();
   if( inWarpLaneIdx == 0 ) {
      shared_results[ warpIdx ] = result_;
      shared_arguments[ warpIdx ] = argument_;
   }

   // The first warp performs the remaining reduction
   __syncthreads();
   if( warpIdx == 0 ) {
      ReturnType partial_result = inWarpLaneIdx < warpsCount ? shared_results[ inWarpLaneIdx ] : identity;
      Index partial_argument = inWarpLaneIdx < warpsCount ? shared_arguments[ inWarpLaneIdx ] : 0;
      auto [ final_result, final_argument ] =
         BlockReduce::template warpReduceWithArgument< warpsPerSegment >( reduce, partial_result, partial_argument );
      // Only the first thread in each group has the correct result
      const int groupIdx = inWarpLaneIdx / warpsPerSegment;
      if( inWarpLaneIdx % warpsPerSegment == 0 && groupIdx < segmentsCount && segmentIdx + groupIdx < end ) {
         const Index currentSegmentIdx = segmentIdx + groupIdx;
         bool emptySegment = ( segments.getOffsets()[ currentSegmentIdx ] == segments.getOffsets()[ currentSegmentIdx + 1 ] );
         store( currentSegmentIdx, final_argument, final_result, emptySegment );
      }
   }
#endif
}

template<
   typename Segments,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value,
   int BlockSize = 256 >
__global__
void
reduceSegmentsCSRDynamicGroupingKernelWithArgument(
   int gridIdx,
   const Index threadsPerSegment,
   const Segments segments,
   Index begin,
   Index end,
   Fetch fetch,
   const Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   constexpr Index warpSize = Backend::getWarpSize();
   constexpr Index warpsPerBlock = BlockSize / warpSize;
   constexpr Index none_scheduled = std::numeric_limits< Index >::max();
   __shared__ Index warps_scheduler[ BlockSize ];
   const auto& offsets = segments.getOffsets();

   const Index segmentIdx =
      threadIdx.x < ( BlockSize / threadsPerSegment )
         ? begin + ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * ( BlockSize / threadsPerSegment ) + threadIdx.x
         : none_scheduled;
   bool reduce_segment = ( segmentIdx < end && threadIdx.x < BlockSize / threadsPerSegment );

   // Processing segments larger than BlockSize
   __shared__ Index scheduled_segment[ 1 ];
   ReturnType result = identity;
   Index argument = 0;

   Index segment_size = -1;
   if( reduce_segment ) {
      segment_size = offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
   }

   if( threadIdx.x == 0 )
      scheduled_segment[ 0 ] = none_scheduled;
   __syncthreads();
   while( true ) {
      if( reduce_segment && segment_size > BlockSize ) {
         AtomicOperations< Devices::GPU >::CAS( scheduled_segment[ 0 ], scheduled_segment[ 0 ], segmentIdx );
      }
      __syncthreads();
      if( scheduled_segment[ 0 ] == none_scheduled )
         break;

      Index globalIdx = offsets[ scheduled_segment[ 0 ] ];
      const Index endIdx = offsets[ scheduled_segment[ 0 ] + 1 ];

      Index localIdx = threadIdx.x;
      while( globalIdx < endIdx ) {
         reduce(
            result,
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, scheduled_segment[ 0 ], localIdx, globalIdx ),
            argument,
            localIdx );
         localIdx += BlockSize;
         globalIdx += BlockSize;
      }
      if( segmentIdx == scheduled_segment[ 0 ] ) {
         reduce_segment = false;
         scheduled_segment[ 0 ] = none_scheduled;
      }
      __syncthreads();

      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< BlockSize, Reduction, ReturnType, Index >;
      __shared__ typename BlockReduce::Storage storage;

      auto [ result_, argument_ ] = BlockReduce::reduceWithArgument( reduce, identity, result, argument, storage, threadIdx.x );

      // Write the result
      if( threadIdx.x == 0 ) {
         TNL_ASSERT_LT( scheduled_segment[ 0 ] + 1, offsets.getSize(), "" );
         bool emptySegment = ( offsets[ scheduled_segment[ 0 ] ] == offsets[ scheduled_segment[ 0 ] + 1 ] );
         store( scheduled_segment[ 0 ], argument_, result_, emptySegment );
      }
   }

   // Processing segments smaller than BlockSize and larger the warp size
   __shared__ int active_warps[ 1 ];
   if( threadIdx.x == 0 )
      active_warps[ 0 ] = 0;
   __syncthreads();

   // Each thread owning segment with size larger than warpSize registers for scheduling
   if( reduce_segment && segment_size > warpSize ) {
      warps_scheduler[ AtomicOperations< Devices::GPU >::add( active_warps[ 0 ], 1 ) ] = segmentIdx;
      reduce_segment = false;
   }
   __syncthreads();

   // Now reduce scheduled segments in warps
   Index warp_idx = threadIdx.x / warpSize;

   while( warp_idx < active_warps[ 0 ] ) {
      Index scheduled_segment = warps_scheduler[ warp_idx ];
      Index globalIdx = offsets[ scheduled_segment ] + ( threadIdx.x & ( warpSize - 1 ) );  // & is cheaper than %
      const Index endIdx = offsets[ scheduled_segment + 1 ];
      result = identity;
      argument = 0;

      Index localIdx = threadIdx.x & ( warpSize - 1 );  // & is cheaper than %
      for( ; globalIdx < endIdx; globalIdx += warpSize ) {
         reduce(
            result,
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, scheduled_segment, localIdx, globalIdx ),
            argument,
            localIdx );
         localIdx += warpSize;
      }
      __syncthreads();

      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< BlockSize, Reduction, ReturnType, Index >;
      auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduce, result, argument );

      // Write the result
      if( ( threadIdx.x & ( warpSize - 1 ) ) == 0 ) {  // first lane in the warp
         TNL_ASSERT_LT( scheduled_segment + 1, offsets.getSize(), "" );
         bool emptySegment = ( offsets[ scheduled_segment ] == offsets[ scheduled_segment + 1 ] );
         store( scheduled_segment, argument_, result_, emptySegment );
      }
      warp_idx += warpsPerBlock;
   }

   // Processing segments smaller than or equal to warp size
   if( reduce_segment ) {
      Index globalIdx = offsets[ segmentIdx ];
      const Index endIdx = offsets[ segmentIdx + 1 ];
      result = identity;
      argument = 0;
      Index localIdx = 0;
      for( ; globalIdx < endIdx; globalIdx++ ) {
         reduce(
            result,
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
            argument,
            localIdx );
         localIdx++;
      }
      // Write the result
      bool emptySegment = ( offsets[ segmentIdx ] == endIdx );
      store( segmentIdx, argument, result, emptySegment );
   }
#endif
}

// Reduction with segment indexes and argument

// TODO: The following vector kernel is special case of the general variable vector kernel.
// Check the performance and if it is the same, we can erase this kernel.
template<
   typename Segments,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegmentsCSRVectorKernelWithIndexesAndArgument(
   Index gridIdx,
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   const Reduction reduction,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   // We map one warp to each segment
   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridIdx ) / Backend::getWarpSize();
   if( segmentIdx_idx >= segmentIndexes.getSize() )
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
      reduction(
         result,
         detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
         argument,
         localIdx );
      localIdx += Backend::getWarpSize();
   }
   // Reduction in each warp which means in each segment.
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduction, result, argument );

   // Write the result
   if( laneIdx == 0 ) {
      bool emptySegment = ( segments.getOffsets()[ segmentIdx ] == endIdx );
      store( segmentIdx_idx, segmentIdx, argument_, result_, emptySegment );
   }

#endif
}

template<
   int ThreadsPerSegment,
   typename Segments,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduce,
   typename Store,
   typename Value >
__global__
void
reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument(
   const Index gridID,
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   Reduce reduce,
   Store store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx_idx =
      ( ( gridID * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( segmentIdx_idx >= segmentIndexes.getSize() )
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
      reduce(
         result,
         detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
         argument,
         localIdx );
      localIdx += ThreadsPerSegment;
   }

   // Parallel reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduce, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::template warpReduceWithArgument< ThreadsPerSegment >( reduce, result, argument );

   // Write the result
   if( laneID == 0 ) {
      TNL_ASSERT_LT( segmentIdx + 1, segments.getOffsets().getSize(), "" );
      bool emptySegment = ( segments.getOffsets()[ segmentIdx ] == segments.getOffsets()[ segmentIdx + 1 ] );
      store( segmentIdx_idx, segmentIdx, argument_, result_, emptySegment );
   }
#endif
}

template<
   int BlockSize,
   int ThreadsPerSegment,
   typename Segments,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegmentsCSRLightMultivectorKernelWithIndexesAndArgument(
   int gridIdx,
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   const Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment;
   if( segmentIdx_idx >= segmentIndexes.getSize() )
      return;

   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   const Index beginIdx = segments.getOffsets()[ segmentIdx ];
   const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

   ReturnType result = identity;
   Index argument = 0;
   Index localIdx = laneIdx;
   for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
      reduce(
         result,
         detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
         argument,
         localIdx );
      localIdx += ThreadsPerSegment;
   }

   // Parallel reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< BlockSize, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduce, result, argument );

   constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
   constexpr int warpsCount = BlockSize / Backend::getWarpSize();
   constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   __shared__ ReturnType shared_results[ warpsCount ];
   __shared__ Index shared_arguments[ warpsCount ];

   // Write results of parallel reduction to shared memory
   __syncthreads();
   if( inWarpLaneIdx == 0 ) {
      shared_results[ warpIdx ] = result_;
      shared_arguments[ warpIdx ] = argument_;
   }

   // The first warp performs the remaining reduction
   __syncthreads();
   if( warpIdx == 0 ) {
      ReturnType partial_result = inWarpLaneIdx < warpsCount ? shared_results[ inWarpLaneIdx ] : identity;
      Index partial_argument = inWarpLaneIdx < warpsCount ? shared_arguments[ inWarpLaneIdx ] : 0;
      auto [ final_result, final_argument ] =
         BlockReduce::template warpReduceWithArgument< warpsPerSegment >( reduce, partial_result, partial_argument );
      // Only the first thread in each group has the correct result
      const int groupIdx = inWarpLaneIdx / warpsPerSegment;
      if( inWarpLaneIdx % warpsPerSegment == 0 && groupIdx < segmentsCount
          && segmentIdx_idx + groupIdx < segmentIndexes.getSize() )
      {
         const Index currentSegmentIdx = segmentIndexes[ segmentIdx_idx + groupIdx ];
         bool emptySegment = ( segments.getOffsets()[ currentSegmentIdx ] == segments.getOffsets()[ currentSegmentIdx + 1 ] );
         store( segmentIdx_idx, segmentIndexes[ segmentIdx_idx + groupIdx ], final_argument, final_result, emptySegment );
      }
   }
#endif
}

template<
   typename Segments,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value,
   int BlockSize = 256 >
__global__
void
reduceSegmentsCSRDynamicGroupingKernelWithIndexesAndArgument(
   int gridIdx,
   const Index threadsPerSegment,
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   const Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   constexpr Index warpSize = Backend::getWarpSize();
   constexpr Index warpsPerBlock = BlockSize / warpSize;
   constexpr Index none_scheduled = std::numeric_limits< Index >::max();
   __shared__ Index warps_scheduler[ BlockSize ];
   const auto& offsets = segments.getOffsets();

   const Index segmentIdx_idx =
      threadIdx.x < ( BlockSize / threadsPerSegment )
         ? ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * ( BlockSize / threadsPerSegment ) + threadIdx.x
         : none_scheduled;
   bool reduce_segment = ( segmentIdx_idx < segmentIndexes.getSize() && threadIdx.x < BlockSize / threadsPerSegment );

   // Processing segments larger than BlockSize
   __shared__ Index scheduled_segment_idx[ 1 ];
   ReturnType result = identity;
   Index argument = 0;

   Index segment_size = -1;
   Index segmentIdx = -1;
   if( reduce_segment ) {
      segmentIdx = segmentIndexes[ segmentIdx_idx ];
      segment_size = offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
   }

   if( threadIdx.x == 0 )
      scheduled_segment_idx[ 0 ] = none_scheduled;
   __syncthreads();
   while( true ) {
      if( reduce_segment && segment_size > BlockSize ) {
         AtomicOperations< Devices::GPU >::CAS( scheduled_segment_idx[ 0 ], scheduled_segment_idx[ 0 ], segmentIdx_idx );
      }
      __syncthreads();
      if( scheduled_segment_idx[ 0 ] == none_scheduled )
         break;

      TNL_ASSERT_LT( scheduled_segment_idx[ 0 ], segmentIndexes.getSize(), "" );
      Index scheduled_segment = segmentIndexes[ scheduled_segment_idx[ 0 ] ];
      Index globalIdx = offsets[ scheduled_segment ];
      const Index endIdx = offsets[ scheduled_segment + 1 ];

      Index localIdx = threadIdx.x;
      while( globalIdx < endIdx ) {
         reduce(
            result,
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, scheduled_segment, localIdx, globalIdx ),
            argument,
            localIdx );
         localIdx += BlockSize;
         globalIdx += BlockSize;
      }
      if( segmentIdx_idx == scheduled_segment_idx[ 0 ] ) {
         reduce_segment = false;
         scheduled_segment_idx[ 0 ] = none_scheduled;
      }
      __syncthreads();

      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< BlockSize, Reduction, ReturnType, Index >;
      __shared__ typename BlockReduce::Storage storage;

      auto [ result_, argument_ ] = BlockReduce::reduceWithArgument( reduce, identity, result, argument, storage, threadIdx.x );

      // Write the result
      if( threadIdx.x == 0 ) {
         TNL_ASSERT_LT( scheduled_segment + 1, offsets.getSize(), "" );
         bool emptySegment = ( offsets[ scheduled_segment ] == offsets[ scheduled_segment + 1 ] );
         store( scheduled_segment_idx[ 0 ], scheduled_segment, argument_, result_, emptySegment );
      }
   }

   // Processing segments smaller than BlockSize and larger the warp size
   __shared__ int active_warps[ 1 ];
   if( threadIdx.x == 0 )
      active_warps[ 0 ] = 0;
   __syncthreads();

   // Each thread owning segment with size larger than warpSize registers for scheduling
   if( reduce_segment && segment_size > warpSize ) {
      warps_scheduler[ AtomicOperations< Devices::GPU >::add( active_warps[ 0 ], 1 ) ] = segmentIdx_idx;
      reduce_segment = false;
   }
   __syncthreads();

   // Now reduce scheduled segments in warps
   Index warp_idx = threadIdx.x / warpSize;

   while( warp_idx < active_warps[ 0 ] ) {
      TNL_ASSERT_LT( warps_scheduler[ warp_idx ], segmentIndexes.getSize(), "" );

      Index scheduled_segment = segmentIndexes[ warps_scheduler[ warp_idx ] ];
      Index globalIdx = offsets[ scheduled_segment ] + ( threadIdx.x & ( warpSize - 1 ) );  // & is cheaper than %
      const Index endIdx = offsets[ scheduled_segment + 1 ];
      result = identity;
      argument = 0;

      Index localIdx = threadIdx.x & ( warpSize - 1 );  // & is cheaper than %
      for( ; globalIdx < endIdx; globalIdx += warpSize ) {
         reduce(
            result,
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, scheduled_segment, localIdx, globalIdx ),
            argument,
            localIdx );
         localIdx += warpSize;
      }
      __syncthreads();

      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< BlockSize, Reduction, ReturnType, Index >;
      auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduce, result, argument );

      // Write the result
      if( ( threadIdx.x & ( warpSize - 1 ) ) == 0 ) {  // first lane in the warp
         TNL_ASSERT_LT( scheduled_segment + 1, offsets.getSize(), "" );
         bool emptySegment = ( offsets[ scheduled_segment ] == offsets[ scheduled_segment + 1 ] );
         store( scheduled_segment_idx[ 0 ], scheduled_segment, argument_, result_, emptySegment );
      }
      warp_idx += warpsPerBlock;
   }

   // Processing segments smaller than or equal to warp size
   if( reduce_segment ) {
      Index globalIdx = offsets[ segmentIdx ];
      const Index endIdx = offsets[ segmentIdx + 1 ];
      result = identity;
      argument = 0;
      Index localIdx = 0;
      for( ; globalIdx < endIdx; globalIdx++ ) {
         reduce(
            result,
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
            argument,
            localIdx );
         localIdx++;
      }
      // Write the result
      bool emptySegment = ( offsets[ segmentIdx ] == endIdx );
      store( segmentIdx_idx, segmentIdx, argument, result, emptySegment );
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
