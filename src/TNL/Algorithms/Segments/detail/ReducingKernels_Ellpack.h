// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Algorithms::Segments::detail {

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
EllpackCudaReductionKernel( Segments segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch fetch,
                            Reduction reduction,
                            ResultKeeper keep,
                            const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   constexpr int warpSize = Backend::getWarpSize();
   const int gridIdx = 0;
   const Index segmentIdx =
      begin + ( ( gridIdx * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / warpSize;
   if( segmentIdx >= end )
      return;

   const Index segmentSize = segments.getSegmentSize();
   ReturnType result = identity;
   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %

   begin = segmentIdx * segmentSize;  // reusing begin and end variables - now they define
   end = begin + segmentSize;         // the range of the global indices

   // Calculate the result
   if constexpr( argumentCount< Fetch >() == 3 ) {
      Index localIdx = laneIdx;
      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += warpSize, localIdx += warpSize ) {
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         result = reduction( result, fetch( segmentIdx, localIdx, globalIdx ) );
      }
   }
   else {
      for( Index i = begin + laneIdx; i < end; i += warpSize )
         result = reduction( result, fetch( i ) );
   }

   // Reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduction, ReturnType >;
   result = BlockReduce::warpReduce( reduction, result );

   // Write the result
   if( laneIdx == 0 )
      keep( segmentIdx, result );

#endif
}

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
EllpackCudaReductionKernelWithArgument( Segments segments,
                                        IndexBegin begin,
                                        IndexEnd end,
                                        Fetch fetch,
                                        Reduction reduction,
                                        ResultKeeper keep,
                                        const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   constexpr int warpSize = Backend::getWarpSize();
   const int gridIdx = 0;
   const Index segmentIdx =
      begin + ( ( gridIdx * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / warpSize;
   if( segmentIdx >= end )
      return;

   const Index segmentSize = segments.getSegmentSize();
   ReturnType result = identity;
   Index argument = 0;
   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %

   begin = segmentIdx * segmentSize;  // reusing begin and end variables - now they define
   end = begin + segmentSize;         // the range of the global indices

   // Calculate the result
   if constexpr( argumentCount< Fetch >() == 3 ) {
      Index localIdx = laneIdx;
      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += warpSize, localIdx += warpSize ) {
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         reduction( result, fetch( segmentIdx, localIdx, globalIdx ), argument, localIdx );
      }
   }
   else {
      for( Index i = begin + laneIdx; i < end; i += warpSize )
         reduction( result, fetch( i ), argument, i - begin );
   }

   // Reduction
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::warpReduceWithArgument( reduction, result, argument );

   // Write the result
   if( laneIdx == 0 )
      keep( segmentIdx, result_, argument_ );

#endif
}

}  // namespace TNL::Algorithms::Segments::detail
