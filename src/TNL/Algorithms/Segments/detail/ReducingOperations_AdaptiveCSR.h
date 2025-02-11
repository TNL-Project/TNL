// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/AdaptiveCSRView.h>
#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>
#include "FetchLambdaAdapter.h"
#include "ReducingOperations_CSR.h"
#include "ReducingKernels_AdaptiveCSR.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index >
struct ReducingOperations< AdaptiveCSRView< Device, Index > >
{
   using SegmentsViewType = AdaptiveCSRView< Device, Index >;
   using ConstViewType = typename SegmentsViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = typename std::remove_const< Index >::type;
   using ConstOffsetsView = typename SegmentsViewType::ConstOffsetsView;
   using CSRViewType = CSRView< Device, Index >;
   using ReducingOperationsCSR = ReducingOperations< CSRViewType >;

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType >
   static void
   reduceSegments( const ConstViewType& segments,
                   IndexBegin begin,
                   IndexEnd end,
                   Fetch fetch,          // TODO Fetch&& fetch does not work here with CUDA
                   Reduction reduction,  // TODO Reduction&& reduction does not work here with CUDA
                   ResultKeeper keeper,  // TODO ResultKeeper&& keeper does not work here with CUDA
                   const Value& identity,
                   const LaunchConfiguration& launchConfig )
   {
      if( std::is_same_v< Device, TNL::Devices::Cuda > || std::is_same_v< Device, TNL::Devices::Hip > ) {
         int valueSizeLog = segments.getSizeValueLog( sizeof( Value ) );
         if( valueSizeLog >= segments.MaxValueSizeLog() ) {
            ReducingOperationsCSR::reduceSegments( segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
            return;
         }

         if constexpr( argumentCount< Fetch >() == 3 ) {
            ReducingOperationsCSR::reduceSegments( segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
         }
         else {
            using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::CudaBlockSize();
            constexpr std::size_t maxGridSize = Backend::getMaxGridXSize();

            // Fill blocks
            const auto blocks = segments.getBlocks()[ valueSizeLog ];
            std::size_t neededThreads = blocks.getSize() * Backend::getWarpSize();  // one warp per block

            // Execute kernels on device
            for( IndexType gridIdx = 0; neededThreads != 0; gridIdx++ ) {
               if( maxGridSize * launch_config.blockSize.x >= neededThreads ) {
                  launch_config.gridSize.x = roundUpDivision( neededThreads, launch_config.blockSize.x );
                  neededThreads = 0;
               }
               else {
                  launch_config.gridSize.x = maxGridSize;
                  neededThreads -= maxGridSize * launch_config.blockSize.x;
               }

               using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
               using BlocksView = typename SegmentsViewType::BlocksView;

               constexpr auto kernel =
                  reduceSegmentsCSRAdaptiveKernel< BlocksView, OffsetsView, IndexType, Fetch, Reduction, ResultKeeper, Value >;
               Backend::launchKernelAsync(
                  kernel, launch_config, gridIdx, blocks, segments.getOffsets(), fetch, reduction, keeper, identity );
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else
         ReducingOperationsCSR::reduceSegments( segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType >
   static void
   reduceSegmentsWithSegmentIndexes( const ConstViewType& segments,
                                     const Array& segmentIndexes,
                                     IndexBegin begin,
                                     IndexEnd end,
                                     Fetch fetch,          // TODO Fetch&& fetch does not work here with CUDA
                                     Reduction reduction,  // TODO Reduction&& reduction does not work here with CUDA
                                     ResultKeeper keeper,  // TODO ResultKeeper&& keeper does not work here with CUDA
                                     const Value& identity,
                                     const LaunchConfiguration& launchConfig )
   {
      ReducingOperationsCSR::reduceSegmentsWithSegmentIndexes(
         segments, segmentIndexes, begin, end, fetch, reduction, keeper, identity, launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType >
   static void
   reduceSegmentsWithArgument( const ConstViewType& segments,
                               IndexBegin begin,
                               IndexEnd end,
                               Fetch fetch,          // TODO Fetch&& fetch does not work here with CUDA
                               Reduction reduction,  // TODO Reduction&& reduction does not work here with CUDA
                               ResultKeeper keeper,  // TODO ResultKeeper&& keeper does not work here with CUDA
                               const Value& identity,
                               const LaunchConfiguration& launchConfig )
   {
      if( std::is_same_v< Device, TNL::Devices::Cuda > || std::is_same_v< Device, TNL::Devices::Hip > ) {
         int valueSizeLog = segments.getSizeValueLog( sizeof( Value ) );
         if( valueSizeLog >= segments.MaxValueSizeLog() ) {
            ReducingOperationsCSR::reduceSegmentsWithArgument(
               segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
            return;
         }

         if constexpr( argumentCount< Fetch >() == 3 ) {
            ReducingOperationsCSR::reduceSegmentsWithArgument(
               segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
         }
         else {
            using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = detail::CSRAdaptiveKernelParameters< sizeof( ReturnType ) >::CudaBlockSize();
            constexpr std::size_t maxGridSize = Backend::getMaxGridXSize();

            // Fill blocks
            const auto& blocks = segments.getBlocks()[ valueSizeLog ];
            std::size_t neededThreads = blocks.getSize() * Backend::getWarpSize();  // one warp per block

            // Execute kernels on device
            for( IndexType gridIdx = 0; neededThreads != 0; gridIdx++ ) {
               if( maxGridSize * launch_config.blockSize.x >= neededThreads ) {
                  launch_config.gridSize.x = roundUpDivision( neededThreads, launch_config.blockSize.x );
                  neededThreads = 0;
               }
               else {
                  launch_config.gridSize.x = maxGridSize;
                  neededThreads -= maxGridSize * launch_config.blockSize.x;
               }

               using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
               using BlocksView = typename SegmentsViewType::BlocksView;
               //OffsetsView offsets = segments.getOffsets();

               constexpr auto kernel = reduceSegmentsCSRAdaptiveKernelWithArgument< BlocksView,
                                                                                    OffsetsView,
                                                                                    IndexType,
                                                                                    Fetch,
                                                                                    Reduction,
                                                                                    ResultKeeper,
                                                                                    Value >;
               Backend::launchKernelAsync(
                  kernel, launch_config, gridIdx, blocks, segments.getOffsets(), fetch, reduction, keeper, identity );
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else
         ReducingOperationsCSR::reduceSegmentsWithArgument(
            segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
   }
};

}  //namespace TNL::Algorithms::Segments::detail
