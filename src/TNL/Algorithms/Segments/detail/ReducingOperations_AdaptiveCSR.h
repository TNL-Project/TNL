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
struct ReducingOperations< AdaptiveCSRView< Device, Index > > : public ReducingOperations< CSRView< Device, Index > >
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
             typename ResultStorer,
             typename Value = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType >
   static void
   reduceSegments( const ConstViewType& segments,
                   IndexBegin begin,
                   IndexEnd end,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultStorer&& storer,
                   const Value& identity,
                   const LaunchConfiguration& launchConfig )
   {
      if( std::is_same_v< Device, TNL::Devices::Cuda > || std::is_same_v< Device, TNL::Devices::Hip > ) {
         int valueSizeLog = segments.getSizeValueLog( sizeof( Value ) );
         if( valueSizeLog >= segments.MaxValueSizeLog() ) {
            ReducingOperationsCSR::reduceSegments( segments, begin, end, fetch, reduction, storer, identity, launchConfig );
            return;
         }

         if constexpr( argumentCount< Fetch >() == 3 ) {
            ReducingOperationsCSR::reduceSegments( segments, begin, end, fetch, reduction, storer, identity, launchConfig );
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

               constexpr auto kernel = reduceSegmentsCSRAdaptiveKernel< BlocksView,
                                                                        OffsetsView,
                                                                        IndexType,
                                                                        std::remove_reference_t< Fetch >,
                                                                        std::remove_reference_t< Reduction >,
                                                                        std::remove_reference_t< ResultStorer >,
                                                                        Value >;
               Backend::launchKernelAsync(
                  kernel, launch_config, gridIdx, blocks, segments.getOffsets(), fetch, reduction, storer, identity );
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else
         ReducingOperationsCSR::reduceSegments( segments, begin, end, fetch, reduction, storer, identity, launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultStorer,
             typename Value = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType >
   static void
   reduceSegmentsWithArgument( const ConstViewType& segments,
                               IndexBegin begin,
                               IndexEnd end,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultStorer&& storer,
                               const Value& identity,
                               const LaunchConfiguration& launchConfig )
   {
      if( std::is_same_v< Device, TNL::Devices::Cuda > || std::is_same_v< Device, TNL::Devices::Hip > ) {
         int valueSizeLog = segments.getSizeValueLog( sizeof( Value ) );
         if( valueSizeLog >= segments.MaxValueSizeLog() ) {
            ReducingOperationsCSR::reduceSegmentsWithArgument(
               segments, begin, end, fetch, reduction, storer, identity, launchConfig );
            return;
         }

         if constexpr( argumentCount< Fetch >() == 3 ) {
            ReducingOperationsCSR::reduceSegmentsWithArgument(
               segments, begin, end, fetch, reduction, storer, identity, launchConfig );
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
                                                                                    std::remove_reference_t< Fetch >,
                                                                                    std::remove_reference_t< Reduction >,
                                                                                    std::remove_reference_t< ResultStorer >,
                                                                                    Value >;
               Backend::launchKernelAsync(
                  kernel, launch_config, gridIdx, blocks, segments.getOffsets(), fetch, reduction, storer, identity );
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else
         ReducingOperationsCSR::reduceSegmentsWithArgument(
            segments, begin, end, fetch, reduction, storer, identity, launchConfig );
   }
};

}  //namespace TNL::Algorithms::Segments::detail
