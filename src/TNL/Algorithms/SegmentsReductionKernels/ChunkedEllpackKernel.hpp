// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/Segments/detail/ChunkedEllpack.h>

#include "ChunkedEllpackKernel.h"
#include "../Segments/detail/ReducingKernels_ChunkedEllpack.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index, typename Device >
template< typename Segments >
void
ChunkedEllpackKernel< Index, Device >::init( const Segments& segments )
{}

template< typename Index, typename Device >
void
ChunkedEllpackKernel< Index, Device >::reset()
{}

template< typename Index, typename Device >
__cuda_callable__
auto
ChunkedEllpackKernel< Index, Device >::getView() -> ViewType
{
   return *this;
}

template< typename Index, typename Device >
__cuda_callable__
auto
ChunkedEllpackKernel< Index, Device >::getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index, typename Device >
std::string
ChunkedEllpackKernel< Index, Device >::getKernelType()
{
   return "ChunkedEllpack";
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
ChunkedEllpackKernel< Index, Device >::reduceSegments( const SegmentsView& segments,
                                                       Index begin,
                                                       Index end,
                                                       Fetch& fetch,
                                                       const Reduction& reduction,
                                                       ResultKeeper& keeper,
                                                       const Value& identity )
{
   using ReturnType = typename Segments::detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   if constexpr( std::is_same_v< DeviceType, Devices::Host > ) {
      for( IndexType segmentIdx = begin; segmentIdx < end; segmentIdx++ ) {
         const IndexType sliceIndex = segments.getSegmentToSliceMappingView()[ segmentIdx ];
         TNL_ASSERT_LE( sliceIndex, segments.getSegmentsCount(), "" );
         IndexType firstChunkOfSegment = 0;
         if( segmentIdx != segments.getSlicesView()[ sliceIndex ].firstSegment )
            firstChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx - 1 ];

         const IndexType lastChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx ];
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = segments.getSlicesView()[ sliceIndex ].pointer;
         const IndexType chunkSize = segments.getSlicesView()[ sliceIndex ].chunkSize;

         ReturnType aux = identity;
         IndexType localIdx = 0;
         if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
            const IndexType segmentSize = segmentChunksCount * chunkSize;
            IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
            IndexType end = begin + segmentSize;
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               aux = reduction(
                  aux,
                  Segments::detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
         }
         else {
            for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
               IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
               IndexType end = begin + segments.getChunksInSlice() * chunkSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += segments.getChunksInSlice() )
                  aux = reduction( aux,
                                   Segments::detail::FetchLambdaAdapter< IndexType, Fetch >::call(
                                      fetch, segmentIdx, localIdx++, globalIdx ) );
            }
         }
         keeper( segmentIdx, aux );
      }
   }
   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
      Backend::LaunchConfiguration launch_config;
      // const IndexType chunksCount = segments.getNumberOfSlices() * segments.getChunksInSlice();
      //  TODO: This ignores parameters begin and end
      const IndexType cudaBlocks = segments.getNumberOfSlices();
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Backend::getMaxGridXSize() );
      launch_config.blockSize.x = segments.getChunksInSlice();
      launch_config.dynamicSharedMemorySize = launch_config.blockSize.x * sizeof( ReturnType );

      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ ) {
         launch_config.gridSize.x = Backend::getMaxGridXSize();
         if( gridIdx == cudaGrids - 1 )
            launch_config.gridSize.x = cudaBlocks % Backend::getMaxGridXSize();
         using ConstSegmentsView = typename SegmentsView::ConstViewType;
         constexpr auto kernel = Segments::detail::
            ChunkedEllpackReduceSegmentsKernel< ConstSegmentsView, IndexType, Fetch, Reduction, ResultKeeper, Value >;
         Backend::launchKernelAsync(
            kernel, launch_config, segments.getConstView(), gridIdx, begin, end, fetch, reduction, keeper, identity );
      }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
ChunkedEllpackKernel< Index, Device >::reduceAllSegments( const SegmentsView& segments,
                                                          Fetch& fetch,
                                                          const Reduction& reduction,
                                                          ResultKeeper& keeper,
                                                          const Value& identity )
{
   reduceSegments( segments, 0, segments.getSegmentsCount(), fetch, reduction, keeper, identity );
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
