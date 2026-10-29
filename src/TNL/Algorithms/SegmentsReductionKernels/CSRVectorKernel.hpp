// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Algorithms/detail/CudaReductionKernel.h>

#include "CSRScalarKernel.h"
#include "CSRVectorKernel.h"
#include "../Segments/detail/ReducingKernels_CSR_Uniform.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

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
std::string
CSRVectorKernel< Index, Device >::getKernelType()
{
   return "Vector";
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
CSRVectorKernel< Index, Device >::reduceSegments(
   const SegmentsView& segments,
   Index begin,
   Index end,
   Fetch& fetch,
   const Reduction& reduction,
   ResultKeeper& keeper,
   const Value& identity )
{
   constexpr bool DispatchScalarCSR = std::is_same_v< Device, Devices::Host >;
   if constexpr( DispatchScalarCSR ) {
      TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel< Index, Device >::reduceSegments(
         segments, begin, end, fetch, reduction, keeper, identity );
   }
   else {
      if( end <= begin )
         return;

      // one warp of the smallest warp size per segment
      constexpr int ThreadsPerSegment = Backend::getMinWarpSize();
      const auto segmentsView = segments.getConstView();
      const std::size_t threadsCount = static_cast< std::size_t >( end - begin ) * ThreadsPerSegment;
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      dim3 blocksCount;
      dim3 gridsCount;
      Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
      for( Index gridIdx = 0; gridIdx < (Index) gridsCount.x; gridIdx++ ) {
         Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
         constexpr auto kernel = Segments::detail::reduceSegments_CSR_Uniform<
            ThreadsPerSegment,
            std::decay_t< decltype( segmentsView ) >,
            Index,
            Fetch,
            Reduction,
            ResultKeeper,
            Value >;
         Backend::launchKernelAsync(
            kernel, launch_config, gridIdx, segmentsView, begin, end, fetch, reduction, keeper, identity );
      }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
CSRVectorKernel< Index, Device >::reduceAllSegments(
   const SegmentsView& segments,
   Fetch& fetch,
   const Reduction& reduction,
   ResultKeeper& keeper,
   const Value& identity )
{
   reduceSegments( segments, 0, segments.getSegmentCount(), fetch, reduction, keeper, identity );
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
