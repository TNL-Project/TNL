// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Cuda/KernelLaunch.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/Segments/detail/ChunkedEllpack.h>

#include "ChunkedEllpackKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename SegmentsView, typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
__global__
void
ChunkedEllpackReduceSegmentsKernel( SegmentsView segments,
                                    Index gridIdx,
                                    Index begin,
                                    Index end,
                                    Fetch fetch,
                                    Reduction reduction,
                                    ResultKeeper keeper,
                                    Value identity )
{
#ifdef __CUDACC__
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index firstSlice = segments.getRowToSliceMappingView()[ begin ];
   const Index lastSlice = segments.getRowToSliceMappingView()[ end - 1 ];

   const Index sliceIdx = firstSlice + gridIdx * Backend::getMaxGridXSize() + blockIdx.x;
   if( sliceIdx > lastSlice )
      return;

   ReturnType* chunksResults = Backend::getSharedMemory< ReturnType >();
   __shared__ Segments::detail::ChunkedEllpackSliceInfo< Index > sliceInfo;

   if( threadIdx.x == 0 )
      sliceInfo = segments.getSlicesView()[ sliceIdx ];
   chunksResults[ threadIdx.x ] = identity;
   __syncthreads();

   const Index sliceOffset = sliceInfo.pointer;
   const Index chunkSize = sliceInfo.chunkSize;
   bool compute = true;

   if constexpr( detail::CheckFetchLambda< Index, Fetch >::hasAllParameters() ) {
      const Index chunkIdx = sliceIdx * segments.getChunksInSlice() + threadIdx.x;
      const Index segmentIdx = segments.getChunksToSegmentsMappingView()[ chunkIdx ];
      Index firstChunkOfSegment = 0;
      if( segmentIdx != sliceInfo.firstSegment )
         firstChunkOfSegment = segments.getRowToChunkMappingView()[ segmentIdx - 1 ];
      Index localIdx = ( threadIdx.x - firstChunkOfSegment ) * chunkSize;

      if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
         Index begin = sliceOffset + threadIdx.x * chunkSize;  // threadIdx.x = chunkIdx within the slice
         Index end = begin + chunkSize;
         for( Index j = begin; j < end && compute; j++ )
            chunksResults[ threadIdx.x ] =
               reduction( chunksResults[ threadIdx.x ], fetch( segmentIdx, localIdx++, j, compute ) );
      }
      else {
         const Index begin = sliceOffset + threadIdx.x;  // threadIdx.x = chunkIdx within the slice
         const Index end = begin + segments.getChunksInSlice() * chunkSize;
         for( Index j = begin; j < end && compute; j += segments.getChunksInSlice() )
            chunksResults[ threadIdx.x ] =
               reduction( chunksResults[ threadIdx.x ], fetch( segmentIdx, localIdx++, j, compute ) );
      }
   }
   else {
      if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
         Index begin = sliceOffset + threadIdx.x * chunkSize;  // threadIdx.x = chunkIdx within the slice
         Index end = begin + chunkSize;
         for( Index j = begin; j < end && compute; j++ )
            chunksResults[ threadIdx.x ] = reduction( chunksResults[ threadIdx.x ], fetch( j, compute ) );
      }
      else {
         const Index begin = sliceOffset + threadIdx.x;  // threadIdx.x = chunkIdx within the slice
         const Index end = begin + segments.getChunksInSlice() * chunkSize;
         for( Index j = begin; j < end && compute; j += segments.getChunksInSlice() )
            chunksResults[ threadIdx.x ] = reduction( chunksResults[ threadIdx.x ], fetch( j, compute ) );
      }
   }

   __syncthreads();

   if( threadIdx.x < sliceInfo.size ) {
      const Index row = sliceInfo.firstSegment + threadIdx.x;
      Index chunkIndex = 0;
      if( threadIdx.x != 0 )
         chunkIndex = segments.getRowToChunkMappingView()[ row - 1 ];
      const Index lastChunk = segments.getRowToChunkMappingView()[ row ];
      ReturnType result = identity;
      while( chunkIndex < lastChunk )
         result = reduction( result, chunksResults[ chunkIndex++ ] );
      if( row >= begin && row < end )
         keeper( row, result );
   }
#endif
}

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
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   if constexpr( std::is_same< DeviceType, Devices::Host >::value ) {
      for( IndexType segmentIdx = begin; segmentIdx < end; segmentIdx++ ) {
         const IndexType sliceIndex = segments.getRowToSliceMappingView()[ segmentIdx ];
         TNL_ASSERT_LE( sliceIndex, segments.getSegmentsCount(), "" );
         IndexType firstChunkOfSegment = 0;
         if( segmentIdx != segments.getSlicesView()[ sliceIndex ].firstSegment )
            firstChunkOfSegment = segments.getRowToChunkMappingView()[ segmentIdx - 1 ];

         const IndexType lastChunkOfSegment = segments.getRowToChunkMappingView()[ segmentIdx ];
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = segments.getSlicesView()[ sliceIndex ].pointer;
         const IndexType chunkSize = segments.getSlicesView()[ sliceIndex ].chunkSize;

         ReturnType aux = identity;
         IndexType localIdx = 0;
         bool compute = true;
         if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
            const IndexType segmentSize = segmentChunksCount * chunkSize;
            IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
            IndexType end = begin + segmentSize;
            for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++ )
               aux = reduction(
                  aux,
                  detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
         }
         else {
            for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
               IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
               IndexType end = begin + segments.getChunksInSlice() * chunkSize;
               for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx += segments.getChunksInSlice() )
                  aux = reduction( aux,
                                   detail::FetchLambdaAdapter< IndexType, Fetch >::call(
                                      fetch, segmentIdx, localIdx++, globalIdx, compute ) );
            }
         }
         keeper( segmentIdx, aux );
      }
   }
   if constexpr( std::is_same< DeviceType, Devices::Cuda >::value ) {
      Devices::Cuda::LaunchConfiguration launch_config;
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
         constexpr auto kernel =
            ChunkedEllpackReduceSegmentsKernel< ConstSegmentsView, IndexType, Fetch, Reduction, ResultKeeper, Value >;
         Cuda::launchKernelAsync(
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
