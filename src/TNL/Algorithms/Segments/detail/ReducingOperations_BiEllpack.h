// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/BiEllpackView.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "FetchLambdaAdapter.h"
#include "ReducingKernels_BiEllpack.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization >
struct ReducingOperations< BiEllpackView< Device, Index, Organization > >
{
   using SegmentsViewType = BiEllpackView< Device, Index, Organization >;
   using ConstViewType = typename SegmentsViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = typename std::remove_const< Index >::type;
   using ConstOffsetsView = typename SegmentsViewType::ConstOffsetsView;

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegments( const ConstViewType& segments,
                   IndexBegin begin,
                   IndexEnd end,
                   Fetch fetch,          // TODO Fetch&& does not work with nvcc
                   Reduction reduction,  // TODO Reduction&& does not work with nvcc
                   ResultKeeper keeper,  // TODO ResultKeeper&& does not work with nvcc
                   const Value& identity,
                   const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      if constexpr( std::is_same_v< DeviceType, Devices::Host > ) {
         for( IndexType segmentIdx = 0; segmentIdx < segments.getSegmentsCount(); segmentIdx++ ) {
            const IndexType stripIdx = segmentIdx / SegmentsViewType::getWarpSize();
            const IndexType groupIdx = stripIdx * ( SegmentsViewType::getLogWarpSize() + 1 );
            const IndexType inStripIdx =
               segments.getSegmentsPermutationView()[ segmentIdx ] - stripIdx * SegmentsViewType::getWarpSize();
            const IndexType groupsCount = Segments::detail::
               BiEllpack< IndexType, DeviceType, SegmentsViewType::getOrganization(), SegmentsViewType::getWarpSize() >::
                  getActiveGroupsCount( segments.getSegmentsPermutationView(), segmentIdx );
            IndexType globalIdx = segments.getGroupPointersView()[ groupIdx ];
            IndexType groupHeight = SegmentsViewType::getWarpSize();
            IndexType localIdx = 0;
            ReturnType aux = identity;
            for( IndexType group = 0; group < groupsCount; group++ ) {
               const IndexType groupSize =
                  Segments::detail::BiEllpack< IndexType,
                                               DeviceType,
                                               SegmentsViewType::getOrganization(),
                                               SegmentsViewType::getWarpSize() >::getGroupSize( segments.getGroupPointersView(),
                                                                                                stripIdx,
                                                                                                group );
               IndexType groupWidth = groupSize / groupHeight;
               const IndexType globalIdxBack = globalIdx;
               if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder )
                  globalIdx += inStripIdx * groupWidth;
               else
                  globalIdx += inStripIdx;
               for( IndexType j = 0; j < groupWidth; j++ ) {
                  aux = reduction(
                     aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
                  if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder )
                     globalIdx++;
                  else
                     globalIdx += groupHeight;
               }
               globalIdx = globalIdxBack + groupSize;
               groupHeight /= 2;
            }
            keeper( segmentIdx, aux );
         }
      }
      if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
         Backend::LaunchConfiguration launch_config;
         constexpr int BlockDim = 256;
         launch_config.blockSize.x = BlockDim;
         const IndexType stripsCount = roundUpDivision( end - begin, SegmentsViewType::getWarpSize() );
         const IndexType cudaBlocks =
            roundUpDivision( stripsCount * SegmentsViewType::getWarpSize(), launch_config.blockSize.x );
         const IndexType cudaGrids = roundUpDivision( cudaBlocks, Backend::getMaxGridXSize() );
         if( SegmentsViewType::getOrganization() == Segments::ColumnMajorOrder )
            launch_config.dynamicSharedMemorySize = launch_config.blockSize.x * sizeof( ReturnType );

         for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ ) {
            launch_config.gridSize.x = Backend::getMaxGridXSize();
            if( gridIdx == cudaGrids - 1 )
               launch_config.gridSize.x = cudaBlocks % Backend::getMaxGridXSize();
            using ConstSegmentsView = typename SegmentsViewType::ConstViewType;
            constexpr auto kernel =
               BiEllpackreduceSegmentsKernel< ConstSegmentsView, IndexType, Fetch, Reduction, ResultKeeper, Value, BlockDim >;
            Backend::launchKernelAsync(
               kernel, launch_config, segments.getConstView(), gridIdx, begin, end, fetch, reduction, keeper, identity );
         }
         Backend::streamSynchronize( launch_config.stream );
      }
   }
};

}  //namespace TNL::Algorithms::Segments::detail
