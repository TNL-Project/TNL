// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/Segments/detail/BiEllpack.h>

#include "BiEllpackKernel.h"
#include "../Segments/detail/ReducingKernels_BiEllpack.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename SegmentsView,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value,
          int BlockDim >
__global__
void
BiEllpackreduceSegmentsKernel( SegmentsView segments,
                               Index gridIdx,
                               Index begin,
                               Index end,
                               Fetch fetch,
                               Reduction reduction,
                               ResultKeeper keeper,
                               Value identity )
{
   if constexpr( Segments::detail::CheckFetchLambda< Index, Fetch >::hasAllParameters() )
      Segments::detail::reduceSegmentsKernelWithAllParameters< BlockDim >(
         segments, gridIdx, begin, end, fetch, reduction, keeper, identity );
   else
      Segments::detail::reduceSegmentsKernel< BlockDim >( segments, gridIdx, begin, end, fetch, reduction, keeper, identity );
}

template< typename Index, typename Device >
template< typename Segments >
void
BiEllpackKernel< Index, Device >::init( const Segments& segments )
{}

template< typename Index, typename Device >
void
BiEllpackKernel< Index, Device >::reset()
{}

template< typename Index, typename Device >
__cuda_callable__
auto
BiEllpackKernel< Index, Device >::getView() -> ViewType
{
   return *this;
}

template< typename Index, typename Device >
__cuda_callable__
auto
BiEllpackKernel< Index, Device >::getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index, typename Device >
std::string
BiEllpackKernel< Index, Device >::getKernelType()
{
   return "BiEllpack";
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
BiEllpackKernel< Index, Device >::reduceSegments( const SegmentsView& segments,
                                                  Index begin,
                                                  Index end,
                                                  Fetch& fetch,
                                                  const Reduction& reduction,
                                                  ResultKeeper& keeper,
                                                  const Value& identity )
{
   using ReturnType = typename Segments::detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   if constexpr( std::is_same_v< DeviceType, Devices::Host > ) {
      for( IndexType segmentIdx = 0; segmentIdx < segments.getSegmentsCount(); segmentIdx++ ) {
         const IndexType stripIdx = segmentIdx / SegmentsView::getWarpSize();
         const IndexType groupIdx = stripIdx * ( SegmentsView::getLogWarpSize() + 1 );
         const IndexType inStripIdx =
            segments.getSegmentsPermutationView()[ segmentIdx ] - stripIdx * SegmentsView::getWarpSize();
         const IndexType groupsCount =
            Segments::detail::BiEllpack< IndexType, DeviceType, SegmentsView::getOrganization(), SegmentsView::getWarpSize() >::
               getActiveGroupsCount( segments.getSegmentsPermutationView(), segmentIdx );
         IndexType globalIdx = segments.getGroupPointersView()[ groupIdx ];
         IndexType groupHeight = SegmentsView::getWarpSize();
         IndexType localIdx = 0;
         ReturnType aux = identity;
         for( IndexType group = 0; group < groupsCount; group++ ) {
            const IndexType groupSize = Segments::detail::
               BiEllpack< IndexType, DeviceType, SegmentsView::getOrganization(), SegmentsView::getWarpSize() >::getGroupSize(
                  segments.getGroupPointersView(), stripIdx, group );
            IndexType groupWidth = groupSize / groupHeight;
            const IndexType globalIdxBack = globalIdx;
            if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder )
               globalIdx += inStripIdx * groupWidth;
            else
               globalIdx += inStripIdx;
            for( IndexType j = 0; j < groupWidth; j++ ) {
               aux = reduction(
                  aux,
                  Segments::detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
               if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder )
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
      const IndexType stripsCount = roundUpDivision( end - begin, SegmentsView::getWarpSize() );
      const IndexType cudaBlocks = roundUpDivision( stripsCount * SegmentsView::getWarpSize(), launch_config.blockSize.x );
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Backend::getMaxGridXSize() );
      if( SegmentsView::getOrganization() == Segments::ColumnMajorOrder )
         launch_config.dynamicSharedMemorySize = launch_config.blockSize.x * sizeof( ReturnType );

      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ ) {
         launch_config.gridSize.x = Backend::getMaxGridXSize();
         if( gridIdx == cudaGrids - 1 )
            launch_config.gridSize.x = cudaBlocks % Backend::getMaxGridXSize();
         using ConstSegmentsView = typename SegmentsView::ConstViewType;
         constexpr auto kernel =
            BiEllpackreduceSegmentsKernel< ConstSegmentsView, IndexType, Fetch, Reduction, ResultKeeper, Value, BlockDim >;
         Backend::launchKernelAsync(
            kernel, launch_config, segments.getConstView(), gridIdx, begin, end, fetch, reduction, keeper, identity );
      }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
BiEllpackKernel< Index, Device >::reduceAllSegments( const SegmentsView& segments,
                                                     Fetch& fetch,
                                                     const Reduction& reduction,
                                                     ResultKeeper& keeper,
                                                     const Value& identity )
{
   reduceSegments( segments, 0, segments.getSegmentsCount(), fetch, reduction, keeper, identity );
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
