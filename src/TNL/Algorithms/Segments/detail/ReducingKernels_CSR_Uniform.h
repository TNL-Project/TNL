// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/detail/CudaReductionKernel.h>
#include <TNL/Algorithms/Segments/detail/FetchLambdaAdapter.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Kernel for the Uniform and UniformWarp thread mappings.
 *
 * A fixed number of threads (\e ThreadsPerSegment) is assigned to every segment regardless of
 * its size, and \e ThreadsPerSegment fits into a single warp, so the partial results can be
 * combined with a single warp-shuffle reduction pass.
 *
 * For segments wide enough to need more than one warp, see \ref reduceSegments_CSR_Uniform_MultipleWarps.
 *
 * The choice of \e ThreadsPerSegment based on the average number of elements per segment is
 * based on the following paper:
 *
 * Y. Liu and B. Schmidt, "LightSpMV: Faster CSR-based sparse matrix-vector multiplication on CUDA-enabled GPUs," 2015 IEEE 26th
 * International Conference on Application-specific Systems, Architectures and Processors (ASAP), Toronto, ON, Canada, 2015, pp.
 * 82-89.
 *
 * \see LaunchConfigurationSetter_LightCSR, which selects \e ThreadsPerSegment for this kernel.
 */
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
reduceSegments_CSR_Uniform(
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

   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridID ) / ThreadsPerSegment;
   const bool active = ( segmentIdx < end );

   ReturnType result = identity;
   if( active ) {
      const Index laneID = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
      Index endID = segments.getOffsets()[ segmentIdx + 1 ];

      // Calculate result
      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         Index localIdx = laneID;
         for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID;
              globalIdx += ThreadsPerSegment )
            result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
         localIdx += ThreadsPerSegment;
      }
      else {
         for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID;
              globalIdx += ThreadsPerSegment )
            result = reduce( result, fetch( globalIdx ) );
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduce, ReturnType >;
   result = BlockReduce::template warpReduce< ThreadsPerSegment >( reduce, result );

   // Write the result
   if( active && ( threadIdx.x & ( ThreadsPerSegment - 1 ) ) == 0 )
      store( segmentIdx, result );
#endif
}

/**
 * \brief Kernel for the Uniform thread mappings when \e ThreadsPerSegment spans
 * more than one warp.
 *
 * This is the counterpart of \ref reduceSegments_CSR_Uniform for segments that need
 * more threads than fit into a single warp (e.g. 64 or 128 threads per segment). Every segment
 * in the launch still gets the same, fixed number of threads (\e ThreadsPerSegment), spanning
 * \e ThreadsPerSegment / warpSize warps. Each warp first reduces its own share with a
 * warp-shuffle pass, the per-warp partial results are written to shared memory, and the first
 * warp of the group performs a second warp-shuffle reduction over them to produce the final
 * result - hence "MultipleWarps": several single-warp reductions combined into one.
 *
 * This is conceptually similar to the \c Type::LONG case of \ref reduceSegmentsCSRAdaptiveKernel,
 * which also lets several warps cooperate on one segment through shared memory. The difference
 * is that here the number of warps per segment is a compile-time constant applied uniformly to
 * every segment processed by this kernel launch (chosen ahead of time from the average segment
 * size, see \ref reduceSegments_CSR_Uniform), whereas AdaptiveCSR computes the number
 * of warps individually for each long segment when building the block descriptors and dedicates
 * a whole thread block to that single segment. The technique of combining several warps through
 * shared memory to reduce one long row was introduced for that AdaptiveCSR-style algorithm by:
 *
 * J. L. Greathouse and M. Daga, "Efficient Sparse Matrix-Vector Multiplication on GPUs Using the
 * CSR Storage Format," in Proceedings of the International Conference for High Performance
 * Computing, Networking, Storage and Analysis (SC '14), 2014.
 */
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
reduceSegments_CSR_Uniform_MultipleWarps(
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
   const bool active = ( segmentIdx < end );

   ReturnType result = identity;
   if( active ) {
      const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
      const Index beginIdx = segments.getOffsets()[ segmentIdx ];
      const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

      Index localIdx = laneIdx;
      for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
         result = reduce( result, detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ) );
         localIdx += ThreadsPerSegment;
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< BlockSize, Reduction, ReturnType >;
   result = BlockReduce::warpReduce( reduce, result );

   constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
   constexpr int warpsCount = BlockSize / Backend::getWarpSize();
   constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %

   // Complex has a non-trivial default constructor, which HIP rejects for __shared__ variables
   __shared__ Backend::Uninitialized< ReturnType > shared[ warpsCount ];

   // Write results of parallel reduction to shared memory
   __syncthreads();
   if( active && inWarpLaneIdx == 0 )
      shared[ warpIdx ] = result;

   // The first warp performs the remaining reduction
   __syncthreads();
   if( warpIdx == 0 ) {
      ReturnType partial = inWarpLaneIdx < warpsCount ? shared[ inWarpLaneIdx ].get() : identity;
      partial = BlockReduce::template warpReduce< warpsPerSegment >( reduce, partial );
      // Only the first thread in each group has the correct result
      const int groupIdx = inWarpLaneIdx / warpsPerSegment;
      if( inWarpLaneIdx % warpsPerSegment == 0 && groupIdx < segmentsCount && segmentIdx + groupIdx < end )
         store( segmentIdx + groupIdx, partial );
   }
#endif
}

// Reduction with segment indexes

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
reduceSegments_CSR_Uniform_WithIndexes(
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

   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridID ) / ThreadsPerSegment;
   const bool active = ( segmentIdx_idx < segmentIndexes.getSize() );

   ReturnType result = identity;
   if( active ) {
      TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index laneID = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
      Index endID = segments.getOffsets()[ segmentIdx + 1 ];

      // Calculate result
      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         Index localIdx = laneID;
         for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID;
              globalIdx += ThreadsPerSegment )
            result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
         localIdx += ThreadsPerSegment;
      }
      else {
         for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID;
              globalIdx += ThreadsPerSegment )
            result = reduce( result, fetch( globalIdx ) );
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduce, ReturnType >;
   result = BlockReduce::template warpReduce< ThreadsPerSegment >( reduce, result );

   // Write the result
   if( active && ( threadIdx.x & ( ThreadsPerSegment - 1 ) ) == 0 ) {
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      store( segmentIdx_idx, segmentIdx, result );
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
reduceSegments_CSR_Uniform_MultipleWarps_WithIndexes(
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
   const bool active = ( segmentIdx_idx < segmentIndexes.getSize() );

   ReturnType result = identity;
   if( active ) {
      TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
      const Index beginIdx = segments.getOffsets()[ segmentIdx ];
      const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

      Index localIdx = laneIdx;
      for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
         result = reduce( result, detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ) );
         localIdx += ThreadsPerSegment;
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< BlockSize, Reduction, ReturnType >;
   result = BlockReduce::warpReduce( reduce, result );

   constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
   constexpr int warpsCount = BlockSize / Backend::getWarpSize();
   constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %

   // Complex has a non-trivial default constructor, which HIP rejects for __shared__ variables
   __shared__ Backend::Uninitialized< ReturnType > shared[ warpsCount ];

   // Write results of parallel reduction to shared memory
   __syncthreads();
   if( active && inWarpLaneIdx == 0 )
      shared[ warpIdx ] = result;

   // The first warp performs the remaining reduction
   __syncthreads();
   if( warpIdx == 0 ) {
      ReturnType partial = inWarpLaneIdx < warpsCount ? shared[ inWarpLaneIdx ].get() : identity;
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

// Reduction with argument

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
reduceSegments_CSR_Uniform_WithArgument(
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

   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridID ) / ThreadsPerSegment;
   const bool active = ( segmentIdx < end );

   ReturnType result = identity;
   Index argument = 0;
   if( active ) {
      const Index laneID = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
      Index endID = segments.getOffsets()[ segmentIdx + 1 ];

      // Calculate result
      Index localIdx = laneID;
      for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID; globalIdx += ThreadsPerSegment ) {
         reduce(
            result,
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
            argument,
            localIdx );
         localIdx += ThreadsPerSegment;
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduce, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::template warpReduceWithArgument< ThreadsPerSegment >( reduce, result, argument );

   // Write the result
   if( active && ( threadIdx.x & ( ThreadsPerSegment - 1 ) ) == 0 ) {
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
reduceSegments_CSR_Uniform_MultipleWarps_WithArgument(
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
   const bool active = ( segmentIdx < end );

   ReturnType result = identity;
   Index argument = 0;
   if( active ) {
      const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
      const Index beginIdx = segments.getOffsets()[ segmentIdx ];
      const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

      Index localIdx = laneIdx;
      for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
         reduce(
            result,
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
            argument,
            localIdx );
         localIdx += ThreadsPerSegment;
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< BlockSize, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduce, result, argument );

   constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
   constexpr int warpsCount = BlockSize / Backend::getWarpSize();
   constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %

   // Complex has a non-trivial default constructor, which HIP rejects for __shared__ variables
   __shared__ Backend::Uninitialized< ReturnType > shared_results[ warpsCount ];
   __shared__ Index shared_arguments[ warpsCount ];

   // Write results of parallel reduction to shared memory
   __syncthreads();
   if( active && inWarpLaneIdx == 0 ) {
      shared_results[ warpIdx ] = result_;
      shared_arguments[ warpIdx ] = argument_;
   }

   // The first warp performs the remaining reduction
   __syncthreads();
   if( warpIdx == 0 ) {
      ReturnType partial_result = inWarpLaneIdx < warpsCount ? shared_results[ inWarpLaneIdx ].get() : identity;
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

// Reduction with segment indexes and argument

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
reduceSegments_CSR_Uniform_WithIndexesAndArgument(
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

   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridID ) / ThreadsPerSegment;
   const bool active = ( segmentIdx_idx < segmentIndexes.getSize() );

   ReturnType result = identity;
   Index argument = 0;
   if( active ) {
      TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index laneID = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
      Index endID = segments.getOffsets()[ segmentIdx + 1 ];

      // Calculate result
      Index localIdx = laneID;
      for( Index globalIdx = segments.getOffsets()[ segmentIdx ] + laneID; globalIdx < endID; globalIdx += ThreadsPerSegment ) {
         reduce(
            result,
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
            argument,
            localIdx );
         localIdx += ThreadsPerSegment;
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduce, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::template warpReduceWithArgument< ThreadsPerSegment >( reduce, result, argument );

   // Write the result
   if( active && ( threadIdx.x & ( ThreadsPerSegment - 1 ) ) == 0 ) {
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
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
reduceSegments_CSR_Uniform_MultipleWarps_WithIndexesAndArgument(
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
   const bool active = ( segmentIdx_idx < segmentIndexes.getSize() );

   ReturnType result = identity;
   Index argument = 0;
   if( active ) {
      TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
      const Index beginIdx = segments.getOffsets()[ segmentIdx ];
      const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];

      Index localIdx = laneIdx;
      for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
         reduce(
            result,
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
            argument,
            localIdx );
         localIdx += ThreadsPerSegment;
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< BlockSize, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduce, result, argument );

   constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
   constexpr int warpsCount = BlockSize / Backend::getWarpSize();
   constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   const Index inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %

   // Complex has a non-trivial default constructor, which HIP rejects for __shared__ variables
   __shared__ Backend::Uninitialized< ReturnType > shared_results[ warpsCount ];
   __shared__ Index shared_arguments[ warpsCount ];

   // Write results of parallel reduction to shared memory
   __syncthreads();
   if( active && inWarpLaneIdx == 0 ) {
      shared_results[ warpIdx ] = result_;
      shared_arguments[ warpIdx ] = argument_;
   }

   // The first warp performs the remaining reduction
   __syncthreads();
   if( warpIdx == 0 ) {
      ReturnType partial_result = inWarpLaneIdx < warpsCount ? shared_results[ inWarpLaneIdx ].get() : identity;
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

}  // namespace TNL::Algorithms::Segments::detail
