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
CSRAdaptiveKernelView< Index, Device >::reduceSegments( const SegmentsView& segments,
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

   constexpr bool DispatchScalarCSR = ( argumentCount< Fetch >() == 3 || std::is_same_v< Device, Devices::Host > );
   if constexpr( DispatchScalarCSR ) {
      TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel< Index, Device >::reduceSegments(
         segments, begin, end, fetch, reduction, keeper, identity );
   }
   else {
      using ReturnType = typename Segments::detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = Segments::detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::CudaBlockSize();
      constexpr std::size_t maxGridSize = Backend::getMaxGridXSize();

      // Fill blocks
      const auto& blocks = this->blocksArray[ valueSizeLog ];
      std::size_t neededThreads = blocks.getSize() * Backend::getWarpSize();  // one warp per block

      // Execute kernels on device
      for( Index gridIdx = 0; neededThreads != 0; gridIdx++ ) {
         if( maxGridSize * launch_config.blockSize.x >= neededThreads ) {
            launch_config.gridSize.x = roundUpDivision( neededThreads, launch_config.blockSize.x );
            neededThreads = 0;
         }
         else {
            launch_config.gridSize.x = maxGridSize;
            neededThreads -= maxGridSize * launch_config.blockSize.x;
         }

         using OffsetsView = typename SegmentsView::ConstOffsetsView;
         OffsetsView offsets = segments.getOffsets();

         constexpr auto kernel = Segments::detail::
            reduceSegmentsCSRAdaptiveKernel< BlocksView, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Value >;
         Backend::launchKernelAsync( kernel, launch_config, blocks, gridIdx, offsets, fetch, reduction, keeper, identity );
      }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
CSRAdaptiveKernelView< Index, Device >::reduceAllSegments( const SegmentsView& segments,
                                                           Fetch& fetch,
                                                           const Reduction& reduction,
                                                           ResultKeeper& keeper,
                                                           const Value& identity ) const
{
   reduceSegments( segments, 0, segments.getSegmentsCount(), fetch, reduction, keeper, identity );
}

template< typename Index, typename Device >
void
CSRAdaptiveKernelView< Index, Device >::printBlocks( int idx ) const
{
   auto& blocks = this->blocksArray[ idx ];
   for( Index i = 0; i < blocks.getSize(); i++ ) {
      auto block = blocks.getElement( i );
      std::cout << "Block " << i << " : " << block << std::endl;
   }
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
