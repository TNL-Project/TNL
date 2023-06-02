// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Cuda/KernelLaunch.h>
#include <TNL/Cuda/LaunchHelpers.h>

#include "detail/FetchLambdaAdapter.h"
#include "CSRScalarKernel.h"
#include "CSRVectorKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Offsets, typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
__global__
void
reduceSegmentsCSRKernelVector( int gridIdx,
                               const Offsets offsets,
                               Index begin,
                               Index end,
                               Fetch fetch,
                               const Reduction reduce,
                               ResultKeeper keep,
                               const Real zero )
{
#ifdef __CUDACC__
   /***
    * We map one warp to each segment
    */
   const Index segmentIdx = TNL::Cuda::getGlobalThreadIdx_x( gridIdx ) / TNL::Cuda::getWarpSize() + begin;
   if( segmentIdx >= end )
      return;

   const int laneIdx = threadIdx.x & ( TNL::Cuda::getWarpSize() - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx( laneIdx );
   Real aux = zero;
   bool compute( true );
   for( Index globalIdx = offsets[ segmentIdx ] + localIdx; globalIdx < endIdx; globalIdx += TNL::Cuda::getWarpSize() ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      aux = reduce( aux, detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx, compute ) );
      localIdx += TNL::Cuda::getWarpSize();
   }

   /****
    * Reduction in each warp which means in each segment.
    */
   aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 16 ) );
   aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 8 ) );
   aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 4 ) );
   aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 2 ) );
   aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 1 ) );

   if( laneIdx == 0 )
      keep( segmentIdx, aux );
#endif
}

template< typename Index, typename Device >
template< typename Segments >
void
CSRVectorKernel< Index, Device >::init( const Segments& segments )
{}

template< typename Index, typename Device >
void
CSRVectorKernel< Index, Device >::reset()
{}

template< typename Index, typename Device >
__cuda_callable__
auto
CSRVectorKernel< Index, Device >::getView() -> ViewType
{
   return *this;
}

template< typename Index, typename Device >
__cuda_callable__
auto
CSRVectorKernel< Index, Device >::getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index, typename Device >
TNL::String
CSRVectorKernel< Index, Device >::getKernelType()
{
   return "Vector";
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
CSRVectorKernel< Index, Device >::reduceSegments( const SegmentsView& segments,
                                                  Index begin,
                                                  Index end,
                                                  Fetch& fetch,
                                                  const Reduction& reduction,
                                                  ResultKeeper& keeper,
                                                  const Real& zero )
{
   constexpr bool DispatchScalarCSR = std::is_same< Device, Devices::Host >::value;
   if constexpr( DispatchScalarCSR ) {
      TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel< Index, Device >::reduceSegments(
         segments, begin, end, fetch, reduction, keeper, zero );
   }
   else {
      if( end <= begin )
         return;

      using OffsetsView = typename SegmentsView::ConstOffsetsView;
      OffsetsView offsets = segments.getOffsets();

      const Index warpsCount = end - begin;
      const std::size_t threadsCount = warpsCount * TNL::Cuda::getWarpSize();
      Cuda::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      dim3 blocksCount;
      dim3 gridsCount;
      TNL::Cuda::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
      for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
         TNL::Cuda::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
         constexpr auto kernel = reduceSegmentsCSRKernelVector< OffsetsView, IndexType, Fetch, Reduction, ResultKeeper, Real >;
         Cuda::launchKernelAsync( kernel, launch_config, gridIdx, offsets, begin, end, fetch, reduction, keeper, zero );
      }
      cudaStreamSynchronize( launch_config.stream );
      TNL_CHECK_CUDA_DEVICE;
   }
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
CSRVectorKernel< Index, Device >::reduceAllSegments( const SegmentsView& segments,
                                                     Fetch& fetch,
                                                     const Reduction& reduction,
                                                     ResultKeeper& keeper,
                                                     const Real& zero )
{
   reduceSegments( segments, 0, segments.getSegmentsCount(), fetch, reduction, keeper, zero );
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
