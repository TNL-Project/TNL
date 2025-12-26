// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ChunkedEllpackView.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "FetchLambdaAdapter.h"
#include "ReducingKernels_ChunkedEllpack.h"
#include "ReducingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization >
struct ReducingOperations< ChunkedEllpackView< Device, Index, Organization > >
: public ReducingOperationsBase< ChunkedEllpackView< Device, Index, Organization > >
{
   using SegmentsViewType = ChunkedEllpackView< Device, Index, Organization >;
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
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultKeeper&& keeper,
                   const Value& identity,
                   const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
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
            if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
               const IndexType segmentSize = segmentChunksCount * chunkSize;
               IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
               IndexType end = begin + segmentSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  aux = reduction(
                     aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
            }
            else {
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + segments.getChunksInSlice() * chunkSize;
                  for( IndexType globalIdx = begin; globalIdx < end; globalIdx += segments.getChunksInSlice() )
                     aux = reduction(
                        aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
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
            using ConstSegmentsView = typename SegmentsViewType::ConstViewType;
            constexpr auto kernel = ChunkedEllpackReduceSegmentsKernel< ConstSegmentsView,
                                                                        IndexType,
                                                                        std::remove_reference_t< Fetch >,
                                                                        std::remove_reference_t< Reduction >,
                                                                        std::remove_reference_t< ResultKeeper >,
                                                                        Value >;
            Backend::launchKernelAsync(
               kernel, launch_config, segments.getConstView(), gridIdx, begin, end, fetch, reduction, keeper, identity );
         }
         Backend::streamSynchronize( launch_config.stream );
      }
   }

   template< typename Array,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithSegmentIndexes( const ConstViewType& segments,
                                     const Array& segmentIndexes,
                                     Fetch&& fetch,
                                     Reduction&& reduction,
                                     ResultKeeper&& keeper,
                                     const Value& identity,
                                     const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      using ArrayView = typename Array::ConstViewType;
      if constexpr( std::is_same_v< DeviceType, Devices::Host > || std::is_same_v< DeviceType, Devices::Sequential > ) {
         for( IndexType segmentIdx_idx = 0; segmentIdx_idx < segmentIndexes.getSize(); segmentIdx_idx++ ) {
            TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
            const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
            const IndexType sliceIndex = segments.getSegmentToSliceMappingView()[ segmentIdx ];
            TNL_ASSERT_LE( sliceIndex, segments.getSegmentsCount(), "" );
            IndexType firstChunkOfSegment = 0;
            if( segmentIdx != segments.getSlicesView()[ sliceIndex ].firstSegment )
               firstChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx - 1 ];

            const IndexType lastChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx ];
            const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
            const IndexType sliceOffset = segments.getSlicesView()[ sliceIndex ].pointer;
            const IndexType chunkSize = segments.getSlicesView()[ sliceIndex ].chunkSize;

            ReturnType result = identity;
            IndexType localIdx = 0;
            if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
               const IndexType segmentSize = segmentChunksCount * chunkSize;
               IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
               IndexType end = begin + segmentSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  result = reduction(
                     result, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
            }
            else {
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + segments.getChunksInSlice() * chunkSize;
                  for( IndexType globalIdx = begin; globalIdx < end; globalIdx += segments.getChunksInSlice() )
                     result = reduction(
                        result,
                        detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
               }
            }
            keeper( segmentIdx_idx, segmentIdx, result );
         }
      }
      else {
         Backend::LaunchConfiguration launch_config;
         // const IndexType chunksCount = segments.getNumberOfSlices() * segments.getChunksInSlice();
         //  TODO: This ignores parameters the size of segmentIndexes
         const IndexType cudaBlocks = segments.getNumberOfSlices();
         const IndexType cudaGrids = roundUpDivision( cudaBlocks, Backend::getMaxGridXSize() );
         launch_config.blockSize.x = segments.getChunksInSlice();
         launch_config.dynamicSharedMemorySize = launch_config.blockSize.x * sizeof( ReturnType );

         for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ ) {
            launch_config.gridSize.x = Backend::getMaxGridXSize();
            if( gridIdx == cudaGrids - 1 )
               launch_config.gridSize.x = cudaBlocks % Backend::getMaxGridXSize();
            using ConstSegmentsView = typename SegmentsViewType::ConstViewType;
            constexpr auto kernel = ChunkedEllpackReduceSegmentsKernelWithIndexes< ConstSegmentsView,
                                                                                   ArrayView,
                                                                                   IndexType,
                                                                                   std::remove_reference_t< Fetch >,
                                                                                   std::remove_reference_t< Reduction >,
                                                                                   std::remove_reference_t< ResultKeeper >,
                                                                                   Value >;
            Backend::launchKernelAsync( kernel,
                                        launch_config,
                                        segments.getConstView(),
                                        segmentIndexes.getConstView(),
                                        gridIdx,
                                        fetch,
                                        reduction,
                                        keeper,
                                        identity );
         }
         Backend::streamSynchronize( launch_config.stream );
      }
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithArgument( const ConstViewType& segments,
                               IndexBegin begin,
                               IndexEnd end,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultKeeper&& keeper,
                               const Value& identity,
                               const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      if constexpr( std::is_same_v< DeviceType, Devices::Host > || std::is_same_v< DeviceType, Devices::Sequential > ) {
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

            ReturnType result = identity;
            IndexType argument = 0;
            IndexType localIdx = 0;
            if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
               const IndexType segmentSize = segmentChunksCount * chunkSize;
               IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
               IndexType end = begin + segmentSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++, localIdx++ )
                  reduction( result,
                             detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                             argument,
                             localIdx );
            }
            else {
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + segments.getChunksInSlice() * chunkSize;
                  for( IndexType globalIdx = begin; globalIdx < end; globalIdx += segments.getChunksInSlice(), localIdx++ )
                     reduction( result,
                                detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                                argument,
                                localIdx );
               }
            }
            keeper( segmentIdx, argument, result );
         }
      }
      else {
         Backend::LaunchConfiguration launch_config;
         // const IndexType chunksCount = segments.getNumberOfSlices() * segments.getChunksInSlice();
         //  TODO: This ignores parameters begin and end
         const IndexType cudaBlocks = segments.getNumberOfSlices();
         const IndexType cudaGrids = roundUpDivision( cudaBlocks, Backend::getMaxGridXSize() );
         launch_config.blockSize.x = segments.getChunksInSlice();
         launch_config.dynamicSharedMemorySize = launch_config.blockSize.x * ( sizeof( ReturnType ) + sizeof( IndexType ) );

         for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ ) {
            launch_config.gridSize.x = Backend::getMaxGridXSize();
            if( gridIdx == cudaGrids - 1 )
               launch_config.gridSize.x = cudaBlocks % Backend::getMaxGridXSize();
            using ConstSegmentsView = typename SegmentsViewType::ConstViewType;
            constexpr auto kernel = ChunkedEllpackReduceSegmentsKernelWithArgument< ConstSegmentsView,
                                                                                    IndexType,
                                                                                    std::remove_reference_t< Fetch >,
                                                                                    std::remove_reference_t< Reduction >,
                                                                                    std::remove_reference_t< ResultKeeper >,
                                                                                    Value >;
            Backend::launchKernelAsync(
               kernel, launch_config, segments.getConstView(), gridIdx, begin, end, fetch, reduction, keeper, identity );
         }
         Backend::streamSynchronize( launch_config.stream );
      }
   }

   template< typename Array,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithSegmentIndexesAndArgument( const ConstViewType& segments,
                                                const Array& segmentIndexes,
                                                Fetch&& fetch,
                                                Reduction&& reduction,
                                                ResultKeeper&& keeper,
                                                const Value& identity,
                                                const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      using ArrayView = typename Array::ConstViewType;
      if constexpr( std::is_same_v< DeviceType, Devices::Host > || std::is_same_v< DeviceType, Devices::Sequential > ) {
         for( IndexType segmentIdx_idx = 0; segmentIdx_idx < segmentIndexes.getSize(); segmentIdx_idx++ ) {
            TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
            const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
            const IndexType sliceIndex = segments.getSegmentToSliceMappingView()[ segmentIdx ];
            TNL_ASSERT_LE( sliceIndex, segments.getSegmentsCount(), "" );
            IndexType firstChunkOfSegment = 0;
            if( segmentIdx != segments.getSlicesView()[ sliceIndex ].firstSegment )
               firstChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx - 1 ];

            const IndexType lastChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx ];
            const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
            const IndexType sliceOffset = segments.getSlicesView()[ sliceIndex ].pointer;
            const IndexType chunkSize = segments.getSlicesView()[ sliceIndex ].chunkSize;

            ReturnType result = identity;
            IndexType localIdx = 0;
            IndexType argument = 0;
            if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
               const IndexType segmentSize = segmentChunksCount * chunkSize;
               IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
               IndexType end = begin + segmentSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++, localIdx++ )
                  reduction( result,
                             detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                             argument,
                             localIdx );
            }
            else {
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + segments.getChunksInSlice() * chunkSize;
                  for( IndexType globalIdx = begin; globalIdx < end; globalIdx += segments.getChunksInSlice(), localIdx++ )
                     reduction( result,
                                detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                                argument,
                                localIdx );
               }
            }
            keeper( segmentIdx_idx, segmentIdx, argument, result );
         }
      }
      else {
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
            using ConstSegmentsView = typename SegmentsViewType::ConstViewType;
            constexpr auto kernel =
               ChunkedEllpackReduceSegmentsKernelWithIndexesAndArgument< ConstSegmentsView,
                                                                         ArrayView,
                                                                         IndexType,
                                                                         std::remove_reference_t< Fetch >,
                                                                         std::remove_reference_t< Reduction >,
                                                                         std::remove_reference_t< ResultKeeper >,
                                                                         Value >;
            Backend::launchKernelAsync( kernel,
                                        launch_config,
                                        segments.getConstView(),
                                        segmentIndexes.getConstView(),
                                        gridIdx,
                                        fetch,
                                        reduction,
                                        keeper,
                                        identity );
         }
         Backend::streamSynchronize( launch_config.stream );
      }
   }
};

}  //namespace TNL::Algorithms::Segments::detail
