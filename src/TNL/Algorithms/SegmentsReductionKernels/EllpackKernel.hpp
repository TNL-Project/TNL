// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/detail/CudaReductionKernel.h>

#include "EllpackKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
__global__
void
EllpackCudaReductionKernel( Index begin,
                            Index end,
                            Fetch fetch,
                            const Reduction reduction,
                            ResultKeeper keep,
                            const Value identity,
                            Index segmentSize )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename Segments::detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   constexpr int warpSize = Backend::getWarpSize();
   const int gridIdx = 0;
   const Index segmentIdx =
      begin + ( ( gridIdx * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / warpSize;
   if( segmentIdx >= end )
      return;

   ReturnType result = identity;
   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   begin = segmentIdx * segmentSize;
   end = begin + segmentSize;

   // Calculate the result
   if constexpr( argumentCount< Fetch >() == 3 ) {
      Index localIdx = laneIdx;
      for( Index i = begin + laneIdx; i < end; i += warpSize, localIdx += warpSize )
         result = reduction( result, fetch( segmentIdx, localIdx, i ) );
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

template< typename Index, typename Device >
template< typename Segments >
void
EllpackKernel< Index, Device >::init( const Segments& segments )
{}

template< typename Index, typename Device >
void
EllpackKernel< Index, Device >::reset()
{}

template< typename Index, typename Device >
__cuda_callable__
auto
EllpackKernel< Index, Device >::getView() -> ViewType
{
   return *this;
}

template< typename Index, typename Device >
__cuda_callable__
auto
EllpackKernel< Index, Device >::getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index, typename Device >
std::string
EllpackKernel< Index, Device >::getKernelType()
{
   return "Ellpack";
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
EllpackKernel< Index, Device >::reduceSegments( const SegmentsView& segments,
                                                Index begin,
                                                Index end,
                                                Fetch& fetch,
                                                const Reduction& reduction,
                                                ResultKeeper& keeper,
                                                const Value& identity )
{
   using ReturnType = typename Segments::detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
      const IndexType segmentSize = segments.getSegmentSize( 0 );
      if constexpr( std::is_same_v< Device, Devices::Cuda > ) {
         if( end <= begin )
            return;
         const Index segmentsCount = end - begin;
         const Index threadsCount = segmentsCount * Backend::getWarpSize();
         const Index blocksCount = Backend::getNumberOfBlocks( threadsCount, 256 );
         Backend::LaunchConfiguration launch_config;
         launch_config.blockSize.x = 256;
         launch_config.gridSize.x = blocksCount;
         constexpr auto kernel = EllpackCudaReductionKernel< IndexType, Fetch, Reduction, ResultKeeper, ReturnType >;
         Backend::launchKernelSync( kernel, launch_config, begin, end, fetch, reduction, keeper, identity, segmentSize );
      }
      else {
         auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
         {
            const IndexType begin = segmentIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            ReturnType aux = identity;
            IndexType localIdx = 0;
            for( IndexType j = begin; j < end; j++ )
               aux = reduction(
                  aux, Segments::detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j ) );
            keeper( segmentIdx, aux );
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }
   else {
      const IndexType storageSize = segments.getStorageSize();
      const IndexType alignedSize = segments.getAlignedSize();
      auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         const IndexType begin = segmentIdx;
         const IndexType end = storageSize;
         ReturnType aux = identity;
         IndexType localIdx = 0;
         for( IndexType j = begin; j < end; j += alignedSize )
            aux = reduction(
               aux, Segments::detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j ) );
         keeper( segmentIdx, aux );
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
EllpackKernel< Index, Device >::reduceAllSegments( const SegmentsView& segments,
                                                   Fetch& fetch,
                                                   const Reduction& reduction,
                                                   ResultKeeper& keeper,
                                                   const Value& identity )
{
   reduceSegments( segments, 0, segments.getSegmentsCount(), fetch, reduction, keeper, identity );
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
