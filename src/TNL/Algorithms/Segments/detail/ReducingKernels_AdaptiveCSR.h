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

      // Stream data to shared memory
      for( Index globalIdx = laneIdx + begin; globalIdx < end; globalIdx += WarpSize )
         streamShared[ warpIdx ][ globalIdx - begin ] = fetch( globalIdx );
      auto warp = cg::tiled_partition< WarpSize >( cg::this_thread_block() );
      warp.sync();
      const Index lastSegmentIdx = firstSegmentIdx + block.getSegmentsInBlock();

      for( Index i = firstSegmentIdx + laneIdx; i < lastSegmentIdx; i += WarpSize ) {
         const Index sharedEnd = offsets[ i + 1 ] - begin;  // end of preprocessed data
         result = identity;
         // Scalar reduction
         for( Index sharedIdx = offsets[ i ] - begin; sharedIdx < sharedEnd; sharedIdx++ )
            result = reduction( result, streamShared[ warpIdx ][ sharedIdx ].get() );
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

      // Stream data to shared memory
      for( Index globalIdx = laneIdx + begin; globalIdx < end; globalIdx += WarpSize )
         streamShared_result[ warpIdx ][ globalIdx - begin ] = fetch( globalIdx );
      auto warp = cg::tiled_partition< WarpSize >( cg::this_thread_block() );
      warp.sync();
      const Index lastSegmentIdx = firstSegmentIdx + block.getSegmentsInBlock();

      for( Index i = firstSegmentIdx + laneIdx; i < lastSegmentIdx; i += WarpSize ) {
         const Index sharedEnd = offsets[ i + 1 ] - begin;  // end of preprocessed data
         result = identity;
         // Scalar reduction
         Index localIdx = 0;
         for( Index sharedIdx = offsets[ i ] - begin; sharedIdx < sharedEnd; sharedIdx++, localIdx++ )
            reduction( result, streamShared_result[ warpIdx ][ sharedIdx ].get(), argument, localIdx );
         bool emptySegment = ( offsets[ i ] == offsets[ i + 1 ] );
         store( i, argument, result, emptySegment );
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
