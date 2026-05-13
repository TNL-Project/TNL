// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/detail/FetchLambdaAdapter.h>
#include <TNL/Algorithms/detail/CudaReductionKernel.h>
#include <TNL/Backend/LaunchHelpers.h>

namespace TNL::Algorithms::Segments::detail {

template<
   int ThreadsPerSegment,
   typename Segments,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
EllpackCudaReductionKernel(
   const Segments segments,
   IndexBegin begin,
   IndexEnd end,
   Fetch fetch,
   Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const int gridIdx = 0;
   const Index segmentIdx =
      begin + ( ( gridIdx * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( segmentIdx >= end )
      return;

   const Index segmentSize = segments.getSegmentSize();
   ReturnType result = identity;
   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %

   begin = segmentIdx * segmentSize;  // reusing begin and end variables - now they define
   end = begin + segmentSize;         // the range of the global indices

   // Calculate the result
   if constexpr( callableArgumentCount< Fetch >() == 3 ) {
      Index localIdx = laneIdx;
      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += ThreadsPerSegment, localIdx += ThreadsPerSegment ) {
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
      }
   }
   else {
      for( Index i = begin + laneIdx; i < end; i += ThreadsPerSegment )
         result = reduce( result, fetch( i ) );
   }

   // Parallel reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduction, ReturnType >;
   result = BlockReduce::warpReduce< ThreadsPerSegment >( reduce, result );

   // Write the result
   if( laneIdx == 0 )
      store( segmentIdx, result );

#endif
}

template<
   int ThreadsPerSegment,
   typename Segments,
   typename ArrayView,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
EllpackCudaReductionKernelWithSegmentIndexes(
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const int gridIdx = 0;
   const Index segmentIdx_idx =
      ( ( gridIdx * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( segmentIdx_idx >= segmentIndexes.getSize() )
      return;

   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   const Index segmentSize = segments.getSegmentSize();
   ReturnType result = identity;
   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %

   Index begin = segmentIdx * segmentSize;
   Index end = begin + segmentSize;

   // Calculate the result
   if constexpr( callableArgumentCount< Fetch >() == 3 ) {
      Index localIdx = laneIdx;
      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += ThreadsPerSegment, localIdx += ThreadsPerSegment ) {
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
      }
   }
   else {
      for( Index i = begin + laneIdx; i < end; i += ThreadsPerSegment )
         result = reduce( result, fetch( i ) );
   }

   // Parallel reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduction, ReturnType >;
   result = BlockReduce::warpReduce< ThreadsPerSegment >( reduce, result );

   // Write the result
   if( laneIdx == 0 )
      store( segmentIdx_idx, segmentIdx, result );

#endif
}

template<
   int ThreadsPerSegment,
   typename Segments,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
EllpackCudaReductionKernelWithArgument(
   const Segments segments,
   IndexBegin begin,
   IndexEnd end,
   Fetch fetch,
   Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const int gridIdx = 0;
   const Index segmentIdx =
      begin + ( ( gridIdx * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( segmentIdx >= end )
      return;

   const Index segmentSize = segments.getSegmentSize();
   ReturnType result = identity;
   Index argument = 0;
   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %

   begin = segmentIdx * segmentSize;  // reusing begin and end variables - now they define
   end = begin + segmentSize;         // the range of the global indices

   // Calculate the result
   if constexpr( callableArgumentCount< Fetch >() == 3 ) {
      Index localIdx = laneIdx;
      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += ThreadsPerSegment, localIdx += ThreadsPerSegment ) {
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         reduce( result, fetch( segmentIdx, localIdx, globalIdx ), argument, localIdx );
      }
   }
   else {
      for( Index i = begin + laneIdx; i < end; i += ThreadsPerSegment )
         reduce( result, fetch( i ), argument, i - begin );
   }

   // Parallel reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument< ThreadsPerSegment >( reduce, result, argument );

   // Write the result
   if( laneIdx == 0 ) {
      bool emptySegment = ( segmentSize == 0 );
      store( segmentIdx, argument_, result_, emptySegment );
   }

#endif
}

template<
   int ThreadsPerSegment,
   typename Segments,
   typename ArrayView,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
EllpackCudaReductionKernelWithSegmentIndexesAndArgument(
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const int gridIdx = 0;
   const Index segmentIdx_idx =
      ( ( gridIdx * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( segmentIdx_idx >= segmentIndexes.getSize() )
      return;

   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   const Index segmentSize = segments.getSegmentSize();
   ReturnType result = identity;
   Index argument = 0;
   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %

   Index begin = segmentIdx * segmentSize;
   Index end = begin + segmentSize;

   // Calculate the result
   Index localIdx = laneIdx;
   for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += ThreadsPerSegment, localIdx += ThreadsPerSegment ) {
      TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
      if constexpr( callableArgumentCount< Fetch >() == 3 )
         reduce( result, fetch( segmentIdx, localIdx, globalIdx ), argument, localIdx );
      else
         reduce( result, fetch( globalIdx ), argument, localIdx );
   }

   // Parallel reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument< ThreadsPerSegment >( reduce, result, argument );

   // Write the result
   if( laneIdx == 0 ) {
      bool emptySegment = ( segmentSize == 0 );
      store( segmentIdx_idx, segmentIdx, argument_, result_, emptySegment );
   }

#endif
}

}  // namespace TNL::Algorithms::Segments::detail
