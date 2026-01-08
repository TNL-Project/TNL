// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Algorithms::Segments::detail {

template< int ThreadsPerSegment,
          typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value >
__global__
void
EllpackCudaReductionKernel( const Segments segments,
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
   if constexpr( argumentCount< Fetch >() == 3 ) {
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
   if( laneIdx == 0 )
      store( segmentIdx, result );

#endif
}

template< int ThreadsPerSegment,
          typename Segments,
          typename ArrayView,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value >
__global__
void
EllpackCudaReductionKernelWithSegmentIndexes( const Segments segments,
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
   if constexpr( argumentCount< Fetch >() == 3 ) {
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
   if( laneIdx == 0 )
      store( segmentIdx_idx, segmentIdx, result );

#endif
}

template< int ThreadsPerSegment,
          typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value >
__global__
void
EllpackCudaReductionKernelWithArgument( const Segments segments,
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
   if constexpr( argumentCount< Fetch >() == 3 ) {
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
   if( laneIdx == 0 ) {
      bool emptySegment = ( segmentSize == 0 );
      store( segmentIdx, argument, result, emptySegment );
   }

#endif
}

template< int ThreadsPerSegment,
          typename Segments,
          typename ArrayView,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value >
__global__
void
EllpackCudaReductionKernelWithSegmentIndexesAndArgument( const Segments segments,
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
      if constexpr( argumentCount< Fetch >() == 3 )
         reduce( result, fetch( segmentIdx, localIdx, globalIdx ), argument, localIdx );
      else
         reduce( result, fetch( globalIdx ), argument, localIdx );
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
   if( laneIdx == 0 ) {
      bool emptySegment = ( segmentSize == 0 );
      store( segmentIdx_idx, segmentIdx, argument, result, emptySegment );
   }

#endif
}

}  // namespace TNL::Algorithms::Segments::detail
