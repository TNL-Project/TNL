// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/detail/CudaReductionKernel.h>
#include <TNL/Algorithms/Segments/detail/FetchLambdaAdapter.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Kernel for the Tiered thread mapping, reducing a contiguous range of segments.
 *
 * Every block initially owns `BlockSize / threadsPerSegment` consecutive segments, one per thread
 * (the remaining threads of the block own no segment and only help with the larger tiers). The
 * segments owned by a block are then processed in three tiers according to their size:
 *
 * 1. Segments larger than the block are processed one at a time by the whole block. The next such
 *    segment is picked via an atomic CAS on a shared variable, its elements are strided over all
 *    threads of the block and the partial results are combined by a block-wide reduction.
 * 2. Segments larger than a warp but not larger than the block are registered in a shared queue via
 *    an atomic counter and the warps of the block take them from the queue one by one. Each segment
 *    is reduced by a single warp using warp shuffles.
 * 3. Segments not larger than a warp are reduced sequentially by the thread that owns them.
 *
 * Unlike \ref reduceSegments_CSR_Uniform, the number of threads working on a segment is thus decided
 * at run time inside the kernel from the actual segment size, which balances the work for segments of
 * very different sizes without any preprocessing. This is the run-time counterpart of the Adaptive
 * mapping of AdaptiveCSR, which assigns the segments to the blocks in advance on the host.
 *
 * The strategy follows the thread/warp/CTA (TWC) gathering of
 *
 * D. Merrill, M. Garland, and A. Grimshaw, "Scalable GPU graph traversal," Proceedings of the 17th ACM SIGPLAN
 * Symposium on Principles and Practice of Parallel Programming (PPoPP '12), New Orleans, LA, USA, 2012, pp. 117-128.
 *
 * with the difference that the smallest segments are processed sequentially by a single thread
 * instead of the scan-based fine-grained gathering used in the paper.
 */
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
reduceSegments_CSR_Tiered(
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
      // Complex has a non-trivial default constructor, which HIP rejects for __shared__ variables
      __shared__ Backend::Uninitialized< typename BlockReduce::Storage > storage;

      result = BlockReduce::reduce( reduce, identity, result, storage.get(), threadIdx.x );

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
      // No __syncthreads() here: the number of iterations of this loop differs between
      // the warps of the block and the warp reduction uses only warp shuffles.
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

/**
 * \brief Kernel for the Tiered thread mapping, reducing the segments given explicitly by
 * \e segmentIndexes.
 *
 * Every thread initially owns one entry of \e segmentIndexes instead of one segment of a contiguous
 * range; otherwise it works the same way as \ref reduceSegments_CSR_Tiered.
 */
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
reduceSegments_CSR_Tiered_WithIndexes(
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

      result = identity;
      TNL_ASSERT_LT( scheduled_segment_idx[ 0 ], segmentIndexes.getSize(), "" );
      Index scheduled_segment = segmentIndexes[ scheduled_segment_idx[ 0 ] ];

      Index globalIdx = offsets[ scheduled_segment ] + threadIdx.x;
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
      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< BlockSize, Reduction, ReturnType >;
      // Complex has a non-trivial default constructor, which HIP rejects for __shared__ variables
      __shared__ Backend::Uninitialized< typename BlockReduce::Storage > storage;

      result = BlockReduce::reduce( reduce, identity, result, storage.get(), threadIdx.x );

      // Write the result
      if( threadIdx.x == 0 )
         store( scheduled_segment_idx[ 0 ], scheduled_segment, result );
      __syncthreads();

      // Mark segment as processed
      if( segmentIdx_idx == scheduled_segment_idx[ 0 ] ) {
         reduce_segment = false;
         scheduled_segment_idx[ 0 ] = none_scheduled;
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
      // No __syncthreads() here: the number of iterations of this loop differs between
      // the warps of the block and the warp reduction uses only warp shuffles.
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

/**
 * \brief Kernel for the Tiered thread mapping, reducing a contiguous range of segments and
 * computing also the position of the result (e.g. the argument of the minimum) within each segment.
 *
 * It works the same way as \ref reduceSegments_CSR_Tiered, only the reduction carries the local
 * index of the element together with the partial result.
 */
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
reduceSegments_CSR_Tiered_WithArgument(
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

      result = identity;
      argument = 0;
      Index globalIdx = offsets[ scheduled_segment[ 0 ] ] + threadIdx.x;
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
      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< BlockSize, Reduction, ReturnType, Index >;
      // Complex has a non-trivial default constructor, which HIP rejects for __shared__ variables
      __shared__ Backend::Uninitialized< typename BlockReduce::Storage > storage;

      auto [ result_, argument_ ] =
         BlockReduce::reduceWithArgument( reduce, identity, result, argument, storage.get(), threadIdx.x );

      // Write the result
      if( threadIdx.x == 0 ) {
         TNL_ASSERT_LT( scheduled_segment[ 0 ] + 1, offsets.getSize(), "" );
         bool emptySegment = ( offsets[ scheduled_segment[ 0 ] ] == offsets[ scheduled_segment[ 0 ] + 1 ] );
         store( scheduled_segment[ 0 ], argument_, result_, emptySegment );
      }
      __syncthreads();

      // Mark segment as processed
      if( segmentIdx == scheduled_segment[ 0 ] ) {
         reduce_segment = false;
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
      // No __syncthreads() here: the number of iterations of this loop differs between
      // the warps of the block and the warp reduction uses only warp shuffles.

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

/**
 * \brief Kernel for the Tiered thread mapping, reducing the segments given explicitly by
 * \e segmentIndexes and computing also the position of the result within each segment.
 *
 * It combines \ref reduceSegments_CSR_Tiered_WithIndexes and
 * \ref reduceSegments_CSR_Tiered_WithArgument.
 */
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
reduceSegments_CSR_Tiered_WithIndexesAndArgument(
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

      result = identity;
      argument = 0;
      TNL_ASSERT_LT( scheduled_segment_idx[ 0 ], segmentIndexes.getSize(), "" );
      Index scheduled_segment = segmentIndexes[ scheduled_segment_idx[ 0 ] ];
      Index globalIdx = offsets[ scheduled_segment ] + threadIdx.x;
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
      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< BlockSize, Reduction, ReturnType, Index >;
      // Complex has a non-trivial default constructor, which HIP rejects for __shared__ variables
      __shared__ Backend::Uninitialized< typename BlockReduce::Storage > storage;

      auto [ result_, argument_ ] =
         BlockReduce::reduceWithArgument( reduce, identity, result, argument, storage.get(), threadIdx.x );

      // Write the result
      if( threadIdx.x == 0 ) {
         TNL_ASSERT_LT( scheduled_segment + 1, offsets.getSize(), "" );
         bool emptySegment = ( offsets[ scheduled_segment ] == offsets[ scheduled_segment + 1 ] );
         store( scheduled_segment_idx[ 0 ], scheduled_segment, argument_, result_, emptySegment );
      }
      __syncthreads();

      // Mark segment as processed
      if( segmentIdx_idx == scheduled_segment_idx[ 0 ] ) {
         reduce_segment = false;
         scheduled_segment_idx[ 0 ] = none_scheduled;
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
      // No __syncthreads() here: the number of iterations of this loop differs between
      // the warps of the block and the warp reduction uses only warp shuffles.

      // Reduction in each warp which means in each segment.
      using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< BlockSize, Reduction, ReturnType, Index >;
      auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduce, result, argument );

      // Write the result
      if( ( threadIdx.x & ( warpSize - 1 ) ) == 0 ) {  // first lane in the warp
         TNL_ASSERT_LT( scheduled_segment + 1, offsets.getSize(), "" );
         bool emptySegment = ( offsets[ scheduled_segment ] == offsets[ scheduled_segment + 1 ] );
         store( warps_scheduler[ warp_idx ], scheduled_segment, argument_, result_, emptySegment );
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
