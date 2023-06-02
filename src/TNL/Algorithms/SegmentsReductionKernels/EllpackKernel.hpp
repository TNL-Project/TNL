// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Cuda/KernelLaunch.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>

#include "detail/FetchLambdaAdapter.h"
#include "EllpackKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
__global__
void
EllpackCudaReductionKernelFull( Index begin,
                                Index end,
                                Fetch fetch,
                                const Reduction reduction,
                                ResultKeeper keep,
                                const Real zero,
                                Index segmentSize )
{
#ifdef __CUDACC__
   const int warpSize = 32;
   const int gridID = 0;
   const Index segmentIdx =
      begin + ( ( gridID * TNL::Cuda::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / warpSize;
   if( segmentIdx >= end )
      return;

   Real result = zero;
   const Index laneID = threadIdx.x & 31;  // & is cheaper than %
   begin = segmentIdx * segmentSize;
   end = begin + segmentSize;

   /* Calculate result */
   Index localIdx( 0 );
   bool compute( true );
   for( Index i = begin + laneID; i < end; i += warpSize )
      result = reduction( result, fetch( segmentIdx, localIdx++, i, compute ) );

   /* Reduction */
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   /* Write result */
   if( laneID == 0 )
      keep( segmentIdx, result );
#endif
}

template< typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
__global__
void
EllpackCudaReductionKernelCompact( Index begin,
                                   Index end,
                                   Fetch fetch,
                                   const Reduction reduction,
                                   ResultKeeper keep,
                                   const Real zero,
                                   Index segmentSize )
{
#ifdef __CUDACC__
   const int warpSize = 32;
   const int gridID = 0;
   const Index segmentIdx =
      begin + ( ( gridID * TNL::Cuda::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / warpSize;
   if( segmentIdx >= end )
      return;

   Real result = zero;
   const Index laneID = threadIdx.x & 31;  // & is cheaper than %
   begin = segmentIdx * segmentSize;
   end = begin + segmentSize;

   /* Calculate result */
   bool compute( true );
   for( Index i = begin + laneID; i < end; i += warpSize )
      result = reduction( result, fetch( i, compute ) );

   /* Reduction */
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   /* Write result */
   if( laneID == 0 )
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
TNL::String
EllpackKernel< Index, Device >::getKernelType()
{
   return "Ellpack";
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
EllpackKernel< Index, Device >::reduceSegments( const SegmentsView& segments,
                                                Index begin,
                                                Index end,
                                                Fetch& fetch,
                                                const Reduction& reduction,
                                                ResultKeeper& keeper,
                                                const Real& zero )
{
   using RealType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
      const IndexType segmentSize = segments.getSegmentSize( 0 );
      if constexpr( std::is_same< Device, Devices::Cuda >::value ) {
         if( end <= begin )
            return;
         const Index segmentsCount = end - begin;
         const Index threadsCount = segmentsCount * 32;
         const Index blocksCount = Cuda::getNumberOfBlocks( threadsCount, 256 );
         Cuda::LaunchConfiguration launch_config;
         launch_config.blockSize.x = 256;
         launch_config.gridSize.x = blocksCount;

         constexpr bool FullFetch = detail::CheckFetchLambda< IndexType, Fetch >::hasAllParameters();
         if constexpr( FullFetch ) {
            constexpr auto kernel = EllpackCudaReductionKernelFull< IndexType, Fetch, Reduction, ResultKeeper, RealType >;
            Cuda::launchKernelSync( kernel, launch_config, begin, end, fetch, reduction, keeper, zero, segmentSize );
         }
         else {
            constexpr auto kernel = EllpackCudaReductionKernelCompact< IndexType, Fetch, Reduction, ResultKeeper, RealType >;
            Cuda::launchKernelSync( kernel, launch_config, begin, end, fetch, reduction, keeper, zero, segmentSize );
         }
      }
      else {
         auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
         {
            const IndexType begin = segmentIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            Real aux( zero );
            IndexType localIdx( 0 );
            bool compute( true );
            for( IndexType j = begin; j < end && compute; j++ )
               aux = reduction(
                  aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j, compute ) );
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
         RealType aux( zero );
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType j = begin; j < end && compute; j += alignedSize )
            aux = reduction(
               aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j, compute ) );
         keeper( segmentIdx, aux );
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
EllpackKernel< Index, Device >::reduceAllSegments( const SegmentsView& segments,
                                                   Fetch& fetch,
                                                   const Reduction& reduction,
                                                   ResultKeeper& keeper,
                                                   const Real& zero )
{
   reduceSegments( segments, 0, segments.getSegmentsCount(), fetch, reduction, keeper, zero );
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
