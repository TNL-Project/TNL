// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Algorithms/detail/CudaReductionKernel.h>

#include "CSRScalarKernel.h"
#include "CSRAdaptiveKernelView.h"
#include "../Segments/detail/ReducingKernels_AdaptiveCSR.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index, typename Device >
void
CSRAdaptiveKernelView< Index, Device >::setBlocks( BlocksType& blocks, const int idx )
{
   this->blocksArray[ idx ].bind( blocks );
}

template< typename Index, typename Device >
__cuda_callable__
auto
CSRAdaptiveKernelView< Index, Device >::getView() -> ViewType
{
   return *this;
}

template< typename Index, typename Device >
__cuda_callable__
auto
CSRAdaptiveKernelView< Index, Device >::getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index, typename Device >
std::string
CSRAdaptiveKernelView< Index, Device >::getKernelType()
{
   return "Adaptive";
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
CSRAdaptiveKernelView< Index, Device >::reduceSegments(
   const SegmentsView& segments,
   Index begin,
   Index end,
   Fetch& fetch,
   const Reduction& reduction,
   ResultKeeper& keeper,
   const Value& identity ) const
{
   int valueSizeLog = getSizeValueLog( sizeof( Value ) );

   if( valueSizeLog >= MaxValueSizeLog ) {
      TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel< Index, Device >::reduceSegments(
         segments, begin, end, fetch, reduction, keeper, identity );
      return;
   }

   constexpr bool DispatchScalarCSR = ( callableArgumentCount< Fetch >() == 3 || std::is_same_v< Device, Devices::Host > );
   if constexpr( DispatchScalarCSR ) {
      TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel< Index, Device >::reduceSegments(
         segments, begin, end, fetch, reduction, keeper, identity );
   }
   else {
      using ReturnType = typename Segments::detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = Segments::detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::CudaBlockSize();
      const std::size_t maxGridSize = Backend::getMaxGridXSize( launch_config.blockSize.x );

      // Fill blocks
      const auto& blocks = this->blocksArray[ valueSizeLog ];
      const std::size_t neededThreads = blocks.getSize() * Backend::getWarpSize( Backend::getDevice() );  // one warp per block
      const std::size_t blocksCount = Backend::getNumberOfBlocks( neededThreads, launch_config.blockSize.x );

      // Execute kernels on device
      for( std::size_t blockOffset = 0; blockOffset < blocksCount; blockOffset += maxGridSize ) {
         launch_config.gridSize.x = Backend::getGridSize( blocksCount, blockOffset, maxGridSize );

         using OffsetsView = typename SegmentsView::ConstOffsetsView;
         OffsetsView offsets = segments.getOffsets();

         constexpr auto kernel = Segments::detail::
            reduceSegmentsCSRAdaptiveKernel< BlocksView, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Value >;
         Backend::launchKernelAsync( kernel, launch_config, blocks, blockOffset, offsets, fetch, reduction, keeper, identity );
      }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
CSRAdaptiveKernelView< Index, Device >::reduceAllSegments(
   const SegmentsView& segments,
   Fetch& fetch,
   const Reduction& reduction,
   ResultKeeper& keeper,
   const Value& identity ) const
{
   reduceSegments( segments, 0, segments.getSegmentCount(), fetch, reduction, keeper, identity );
}

template< typename Index, typename Device >
void
CSRAdaptiveKernelView< Index, Device >::printBlocks( int idx ) const
{
   auto& blocks = this->blocksArray[ idx ];
   for( Index i = 0; i < blocks.getSize(); i++ ) {
      auto block = blocks.getElement( i );
      std::cout << "Block " << i << " : " << block << '\n';
   }
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
