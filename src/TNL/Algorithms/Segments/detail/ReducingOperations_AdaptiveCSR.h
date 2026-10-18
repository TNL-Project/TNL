// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/AdaptiveCSRView.h>
#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>
#include "FetchLambdaAdapter.h"
#include "ReducingOperationsBase.h"
#include "ReducingKernels_AdaptiveCSR.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index >
struct ReducingOperations< AdaptiveCSRView< Device, Index > > : public ReducingOperations< CSRView< Device, Index > >
{
   using SegmentsViewType = AdaptiveCSRView< Device, Index >;
   using ConstViewType = typename SegmentsViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;
   using ConstOffsetsView = typename SegmentsViewType::ConstOffsetsView;
   using CSRViewType = CSRView< Device, Index >;
   using ReducingOperationsCSR = ReducingOperations< CSRViewType >;

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Fetch,
      typename Reduction,
      typename ResultStorer,
      typename Value = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType >
   static void
   reduceSegments(
      const ConstViewType& segments,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      const LaunchConfiguration& launchConfig )
   {
      if constexpr( std::is_same_v< Device, TNL::Devices::GPU > ) {
         // The precomputed adaptive blocks cover the whole matrix, so they cannot be used
         // to restrict the reduction to a sub-range of segments. Fall back to the plain
         // CSR kernels, which respect begin/end, whenever a sub-range is requested.
         int valueSizeLog = segments.getSizeValueLog( sizeof( Value ) );
         if( valueSizeLog >= segments.MaxValueSizeLog() || begin != 0 || end != segments.getSegmentCount() ) {
            ReducingOperationsCSR::reduceSegments( segments, begin, end, fetch, reduction, storer, identity, launchConfig );
            return;
         }

         using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
         Backend::LaunchConfiguration launch_config;
         launch_config.blockSize.x = detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::CudaBlockSize();
         const std::size_t maxGridSize = Backend::getMaxGridXSize( launch_config.blockSize.x );

         // Fill blocks
         const auto blocks = segments.getBlocks()[ valueSizeLog ];
         const std::size_t neededThreads =
            blocks.getSize() * Backend::getWarpSize( Backend::getDevice() );  // one warp per block
         const std::size_t blocksCount = Backend::getNumberOfBlocks( neededThreads, launch_config.blockSize.x );

         // Execute kernels on device
         for( std::size_t blockOffset = 0; blockOffset < blocksCount; blockOffset += maxGridSize ) {
            launch_config.gridSize.x = Backend::getGridSize( blocksCount, blockOffset, maxGridSize );

            using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
            using BlocksView = typename SegmentsViewType::BlocksView;

            constexpr auto kernel = reduceSegmentsCSRAdaptiveKernel<
               BlocksView,
               OffsetsView,
               IndexType,
               std::remove_reference_t< Fetch >,
               std::remove_reference_t< Reduction >,
               std::remove_reference_t< ResultStorer >,
               Value >;
            Backend::launchKernelAsync(
               kernel, launch_config, blockOffset, blocks, segments.getOffsets(), fetch, reduction, storer, identity );
         }
         Backend::streamSynchronize( launch_config.stream );
      }
      else {
         ReducingOperationsCSR::reduceSegments( segments, begin, end, fetch, reduction, storer, identity, launchConfig );
      }
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Fetch,
      typename Reduction,
      typename ResultStorer,
      typename Value = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType >
   static void
   reduceSegmentsWithArgument(
      const ConstViewType& segments,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      const LaunchConfiguration& launchConfig )
   {
      if constexpr( std::is_same_v< Device, TNL::Devices::GPU > ) {
         // The precomputed adaptive blocks cover the whole matrix, so they cannot be used
         // to restrict the reduction to a sub-range of segments. Fall back to the plain
         // CSR kernels, which respect begin/end, whenever a sub-range is requested.
         int valueSizeLog = segments.getSizeValueLog( sizeof( Value ) );
         if( valueSizeLog >= segments.MaxValueSizeLog() || begin != 0 || end != segments.getSegmentCount() ) {
            ReducingOperationsCSR::reduceSegmentsWithArgument(
               segments, begin, end, fetch, reduction, storer, identity, launchConfig );
            return;
         }

         using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
         Backend::LaunchConfiguration launch_config;
         launch_config.blockSize.x = detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::CudaBlockSize();
         const std::size_t maxGridSize = Backend::getMaxGridXSize( launch_config.blockSize.x );

         // Fill blocks
         const auto& blocks = segments.getBlocks()[ valueSizeLog ];
         const std::size_t neededThreads =
            blocks.getSize() * Backend::getWarpSize( Backend::getDevice() );  // one warp per block
         const std::size_t blocksCount = Backend::getNumberOfBlocks( neededThreads, launch_config.blockSize.x );

         // Execute kernels on device
         for( std::size_t blockOffset = 0; blockOffset < blocksCount; blockOffset += maxGridSize ) {
            launch_config.gridSize.x = Backend::getGridSize( blocksCount, blockOffset, maxGridSize );

            using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
            using BlocksView = typename SegmentsViewType::BlocksView;

            constexpr auto kernel = reduceSegmentsCSRAdaptiveKernelWithArgument<
               BlocksView,
               OffsetsView,
               IndexType,
               std::remove_reference_t< Fetch >,
               std::remove_reference_t< Reduction >,
               std::remove_reference_t< ResultStorer >,
               Value >;
            Backend::launchKernelAsync(
               kernel, launch_config, blockOffset, blocks, segments.getOffsets(), fetch, reduction, storer, identity );
         }
         Backend::streamSynchronize( launch_config.stream );
      }
      else {
         ReducingOperationsCSR::reduceSegmentsWithArgument(
            segments, begin, end, fetch, reduction, storer, identity, launchConfig );
      }
   }
};

}  // namespace TNL::Algorithms::Segments::detail
