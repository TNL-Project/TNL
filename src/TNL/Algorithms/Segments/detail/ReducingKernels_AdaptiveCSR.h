// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/detail/CudaReductionKernel.h>
#include <TNL/Algorithms/Segments/detail/CSRAdaptiveKernelBlockDescriptor.h>
#include <TNL/Algorithms/Segments/detail/CSRAdaptiveKernelParameters.h>
#include <TNL/Algorithms/Segments/detail/FetchLambdaAdapter.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Kernels for the AdaptiveCSR format.
 *
 * Each block of warps is dynamically assigned one of three strategies depending on the sizes of
 * the segments it covers, as recorded in the block descriptors built ahead of the kernel launch:
 * - \c Type::STREAM - several short segments handled together by a warp, using shared memory for
 *   coalesced loads.
 * - \c Type::VECTOR - one segment per warp.
 * - \c Type::LONG - a single, very long segment split across several warps of the block, whose
 *   partial results are combined through shared memory.
 *
 * The \c Type::LONG case is conceptually similar to \ref reduceSegments_CSR_Uniform_MultipleWarps
 * (also several warps cooperating on one segment via shared memory), but here the number of
 * warps assigned to a segment is computed individually for each long segment while building the
 * block descriptors, and the whole thread block is dedicated to that single segment - as opposed
 * to a fixed, compile-time thread count applied uniformly to a whole batch of segments.
 *
 * \see J. L. Greathouse and M. Daga, "Efficient Sparse Matrix-Vector Multiplication on GPUs
 *      Using the CSR Storage Format," in Proceedings of the International Conference for High
 *      Performance Computing, Networking, Storage and Analysis (SC '14), 2014.
 * \see M. Daga and J. L. Greathouse, "Structural Agnostic SpMV: Adapting CSR-Adaptive for
 *      Irregular Matrices," in Proceedings of the IEEE 22nd International Conference on High
 *      Performance Computing (HiPC), 2015, pp. 64-74.
 */
template<
   typename BlocksView,
   typename Offsets,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegmentsCSRAdaptiveKernel(
   int gridIdx,
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

   // Complex has a non-trivial default constructor, which HIP rejects for __shared__ variables
   __shared__ Backend::Uninitialized< ReturnType > streamShared[ WarpsCount ][ StreamedSharedElementsPerWarp ];
   __shared__ Backend::Uninitialized< ReturnType > multivectorShared[ CudaBlockSize / WarpSize ];

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
      const Index lastSegmentIdx = firstSegmentIdx + block.getSegmentsInBlock();

      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         // 3-arg fetch: reduce per-segment directly (no shared memory streaming)
         for( Index i = firstSegmentIdx + laneIdx; i < lastSegmentIdx; i += WarpSize ) {
            const Index segBegin = offsets[ i ];
            const Index segEnd = offsets[ i + 1 ];
            result = identity;
            Index localIdx = 0;
            for( Index globalIdx = segBegin; globalIdx < segEnd; globalIdx++, localIdx++ )
               result = reduction( result, FetchLambdaAdapter< Index, Fetch >::call( fetch, i, localIdx, globalIdx ) );
            store( i, result );
         }
      }
      else {
         // 1-arg fetch: stream data to shared memory for coalesced access
         for( Index globalIdx = laneIdx + begin; globalIdx < end; globalIdx += WarpSize )
            streamShared[ warpIdx ][ globalIdx - begin ] = fetch( globalIdx );
         auto warp = cg::tiled_partition< WarpSize >( cg::this_thread_block() );
         warp.sync();

         for( Index i = firstSegmentIdx + laneIdx; i < lastSegmentIdx; i += WarpSize ) {
            const Index sharedEnd = offsets[ i + 1 ] - begin;  // end of preprocessed data
            result = identity;
            // Scalar reduction
            for( Index sharedIdx = offsets[ i ] - begin; sharedIdx < sharedEnd; sharedIdx++ )
               result = reduction( result, streamShared[ warpIdx ][ sharedIdx ].get() );
            store( i, result );
         }
      }
   }
   else if( block.getType() == detail::Type::VECTOR )  // Vector kernel - one segment per warp
   {
      const Index end = begin + block.getSize();
      const Index segmentIdx = block.getFirstSegment();

      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += WarpSize )
         result =
            reduction( result, FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, globalIdx - begin, globalIdx ) );

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
         result =
            reduction( result, FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, globalIdx - begin, globalIdx ) );
      }

      // Parallel reduction
      using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduction, ReturnType >;
      result = BlockReduce::warpReduce( reduction, result );

      const Index warpIdx = threadIdx.x / Backend::getWarpSize();
      if( laneIdx == 0 )
         multivectorShared[ warpIdx ] = result;

      __syncthreads();
      // Reduction in multivectorShared using warp-level shuffle
      if( block.getWarpIdx() == 0 ) {
         constexpr int totalWarps = CudaBlockSize / WarpSize;
         auto myValue = ( laneIdx < totalWarps ) ? multivectorShared[ laneIdx ].get() : identity;
         myValue = BlockReduce::warpReduce( reduction, myValue );
         if( laneIdx == 0 )
            multivectorShared[ 0 ] = myValue;
      }
      __syncthreads();

      if( laneIdx == 0 ) {
         store( segmentIdx, multivectorShared[ 0 ].get() );
      }
   }
#endif
}

template<
   typename BlocksView,
   typename Offsets,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegmentsCSRAdaptiveKernelWithArgument(
   int gridIdx,
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

   // Complex has a non-trivial default constructor, which HIP rejects for __shared__ variables
   __shared__ Backend::Uninitialized< ReturnType > streamShared_result[ WarpsCount ][ StreamedSharedElementsPerWarp ];
   __shared__ Backend::Uninitialized< ReturnType > multivectorShared_result[ CudaBlockSize / WarpSize ];
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
      const Index lastSegmentIdx = firstSegmentIdx + block.getSegmentsInBlock();

      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         // 3-arg fetch: reduce per-segment directly (no shared memory streaming)
         for( Index i = firstSegmentIdx + laneIdx; i < lastSegmentIdx; i += WarpSize ) {
            const Index segBegin = offsets[ i ];
            const Index segEnd = offsets[ i + 1 ];
            result = identity;
            Index localIdx = 0;
            for( Index globalIdx = segBegin; globalIdx < segEnd; globalIdx++, localIdx++ )
               reduction(
                  result, FetchLambdaAdapter< Index, Fetch >::call( fetch, i, localIdx, globalIdx ), argument, localIdx );
            bool emptySegment = ( segBegin == segEnd );
            store( i, argument, result, emptySegment );
         }
      }
      else {
         // 1-arg fetch: stream data to shared memory for coalesced access
         for( Index globalIdx = laneIdx + begin; globalIdx < end; globalIdx += WarpSize )
            streamShared_result[ warpIdx ][ globalIdx - begin ] = fetch( globalIdx );
         auto warp = cg::tiled_partition< WarpSize >( cg::this_thread_block() );
         warp.sync();

         for( Index i = firstSegmentIdx + laneIdx; i < lastSegmentIdx; i += WarpSize ) {
            const Index sharedEnd = offsets[ i + 1 ] - begin;  // end of preprocessed data
            result = identity;
            Index localIdx = 0;
            for( Index sharedIdx = offsets[ i ] - begin; sharedIdx < sharedEnd; sharedIdx++, localIdx++ )
               reduction( result, streamShared_result[ warpIdx ][ sharedIdx ].get(), argument, localIdx );
            bool emptySegment = ( offsets[ i ] == offsets[ i + 1 ] );
            store( i, argument, result, emptySegment );
         }
      }
   }
   else if( block.getType() == detail::Type::VECTOR )  // Vector kernel - one segment per warp
   {
      const Index end = begin + block.getSize();
      const Index segmentIdx = block.getFirstSegment();

      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += WarpSize )
         reduction(
            result,
            FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, globalIdx - begin, globalIdx ),
            argument,
            globalIdx - begin );

      // Parallel reduction
      using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
      auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduction, result, argument );

      if( laneIdx == 0 ) {
         bool emptySegment = ( begin == end );
         store( segmentIdx, argument_, result_, emptySegment );
      }
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
         reduction(
            result,
            FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, globalIdx - begin, globalIdx ),
            argument,
            globalIdx - begin );
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
      // Reduction in multivectorShared using warp-level shuffle
      if( block.getWarpIdx() == 0 ) {
         constexpr int totalWarps = CudaBlockSize / WarpSize;
         auto myResult = ( laneIdx < totalWarps ) ? multivectorShared_result[ laneIdx ].get() : identity;
         auto myArgument = ( laneIdx < totalWarps ) ? multivectorShared_argument[ laneIdx ] : Index{};
         auto [ reducedResult, reducedArgument ] = BlockReduce::warpReduceWithArgument( reduction, myResult, myArgument );
         if( laneIdx == 0 ) {
            multivectorShared_result[ 0 ] = reducedResult;
            multivectorShared_argument[ 0 ] = reducedArgument;
         }
      }
      __syncthreads();

      if( laneIdx == 0 ) {
         bool emptySegment = ( begin == end );
         store( segmentIdx, multivectorShared_argument[ 0 ], multivectorShared_result[ 0 ].get(), emptySegment );
      }
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
