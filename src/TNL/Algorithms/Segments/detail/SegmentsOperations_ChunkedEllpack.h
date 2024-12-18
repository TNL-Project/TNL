// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ChunkedEllpackView.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization >
struct SegmentsOperations< ChunkedEllpackView< Device, Index, Organization > >
{
   using ViewType = Segments::ChunkedEllpackView< Device, Index, Organization >;
   using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = Index;
   using ConstOffsetsView = typename ViewType::ConstOffsetsView;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      const IndexType chunksInSlice = segments.getChunksInSlice();
      auto segmentToChunkMapping = segments.getSegmentToChunkMappingView();
      auto segmentToSliceMapping = segments.getSegmentToSliceMappingView();
      auto slices = segments.getSlicesView();
      if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
         auto work = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
         {
            const IndexType sliceIdx = segmentToSliceMapping[ segmentIdx ];

            IndexType firstChunkOfSegment( 0 );
            if( segmentIdx != slices[ sliceIdx ].firstSegment ) {
               firstChunkOfSegment = segmentToChunkMapping[ segmentIdx - 1 ];
            }

            const IndexType lastChunkOfSegment = segmentToChunkMapping[ segmentIdx ];
            const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
            const IndexType sliceOffset = slices[ sliceIdx ].pointer;
            const IndexType chunkSize = slices[ sliceIdx ].chunkSize;

            const IndexType segmentSize = segmentChunksCount * chunkSize;
            if( Organization == RowMajorOrder ) {
               IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
               IndexType end = begin + segmentSize;
               IndexType localIdx = 0;
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, localIdx++, j );
            }
            else {
               IndexType localIdx = 0;
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + chunksInSlice * chunkSize;
                  for( IndexType j = begin; j < end; j += chunksInSlice ) {
                     function( segmentIdx, localIdx++, j );
                  }
               }
            }
         };
         Algorithms::parallelFor< DeviceType >( begin, end, work );
      }
      else {  // argumentCount< Function >() == 2
         auto work = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
         {
            const IndexType sliceIdx = segmentToSliceMapping[ segmentIdx ];

            IndexType firstChunkOfSegment( 0 );
            if( segmentIdx != slices[ sliceIdx ].firstSegment ) {
               firstChunkOfSegment = segmentToChunkMapping[ segmentIdx - 1 ];
            }

            const IndexType lastChunkOfSegment = segmentToChunkMapping[ segmentIdx ];
            const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
            const IndexType sliceOffset = slices[ sliceIdx ].pointer;
            const IndexType chunkSize = slices[ sliceIdx ].chunkSize;

            const IndexType segmentSize = segmentChunksCount * chunkSize;
            if( Organization == RowMajorOrder ) {
               IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
               IndexType end = begin + segmentSize;
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, j );
            }
            else {
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + chunksInSlice * chunkSize;
                  for( IndexType j = begin; j < end; j += chunksInSlice ) {
                     function( segmentIdx, j );
                  }
               }
            }
         };
         Algorithms::parallelFor< DeviceType >( begin, end, work );
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      return forElements( segments.getConstView(), begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      const IndexType chunksInSlice = segments.getChunksInSlice();
      auto segmentToChunkMapping = segments.getSegmentToChunkMappingView();
      auto segmentToSliceMapping = segments.getSegmentToSliceMappingView();
      auto slices = segments.getSlicesView();
      auto segmentIndexesView = segmentIndexes.getConstView();
      if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
         auto work = [ = ] __cuda_callable__( IndexType idx ) mutable
         {
            const IndexType segmentIdx = segmentIndexesView[ idx ];
            const IndexType sliceIdx = segmentToSliceMapping[ segmentIdx ];

            IndexType firstChunkOfSegment( 0 );
            if( segmentIdx != slices[ sliceIdx ].firstSegment ) {
               firstChunkOfSegment = segmentToChunkMapping[ segmentIdx - 1 ];
            }

            const IndexType lastChunkOfSegment = segmentToChunkMapping[ segmentIdx ];
            const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
            const IndexType sliceOffset = slices[ sliceIdx ].pointer;
            const IndexType chunkSize = slices[ sliceIdx ].chunkSize;

            const IndexType segmentSize = segmentChunksCount * chunkSize;
            if( Organization == RowMajorOrder ) {
               IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
               IndexType end = begin + segmentSize;
               IndexType localIdx = 0;
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, localIdx++, j );
            }
            else {
               IndexType localIdx = 0;
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + chunksInSlice * chunkSize;
                  for( IndexType j = begin; j < end; j += chunksInSlice ) {
                     function( segmentIdx, localIdx++, j );
                  }
               }
            }
         };
         Algorithms::parallelFor< DeviceType >( begin, end, work );
      }
      else {  // argumentCount< Function >() == 2
         auto work = [ = ] __cuda_callable__( IndexType idx ) mutable
         {
            const IndexType segmentIdx = segmentIndexesView[ idx ];
            const IndexType sliceIdx = segmentToSliceMapping[ segmentIdx ];

            IndexType firstChunkOfSegment( 0 );
            if( segmentIdx != slices[ sliceIdx ].firstSegment ) {
               firstChunkOfSegment = segmentToChunkMapping[ segmentIdx - 1 ];
            }

            const IndexType lastChunkOfSegment = segmentToChunkMapping[ segmentIdx ];
            const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
            const IndexType sliceOffset = slices[ sliceIdx ].pointer;
            const IndexType chunkSize = slices[ sliceIdx ].chunkSize;

            const IndexType segmentSize = segmentChunksCount * chunkSize;
            if( Organization == RowMajorOrder ) {
               IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
               IndexType end = begin + segmentSize;
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, j );
            }
            else {
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + chunksInSlice * chunkSize;
                  for( IndexType j = begin; j < end; j += chunksInSlice ) {
                     function( segmentIdx, j );
                  }
               }
            }
         };
         Algorithms::parallelFor< DeviceType >( begin, end, work );
      }
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ViewType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      return forElements(
         segments.getConstView(), segmentIndexes, begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ConstViewType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  const LaunchConfiguration& launchConfig,
                  Condition condition,
                  Function function )
   {
      const IndexType chunksInSlice = segments.getChunksInSlice();
      auto segmentToChunkMapping = segments.getSegmentToChunkMappingView();
      auto segmentToSliceMapping = segments.getSegmentToSliceMappingView();
      auto slices = segments.getSlicesView();
      if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
         auto work = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
         {
            if( ! condition( segmentIdx ) )
               return;
            const IndexType sliceIdx = segmentToSliceMapping[ segmentIdx ];

            IndexType firstChunkOfSegment( 0 );
            if( segmentIdx != slices[ sliceIdx ].firstSegment ) {
               firstChunkOfSegment = segmentToChunkMapping[ segmentIdx - 1 ];
            }

            const IndexType lastChunkOfSegment = segmentToChunkMapping[ segmentIdx ];
            const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
            const IndexType sliceOffset = slices[ sliceIdx ].pointer;
            const IndexType chunkSize = slices[ sliceIdx ].chunkSize;

            const IndexType segmentSize = segmentChunksCount * chunkSize;
            if( Organization == RowMajorOrder ) {
               IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
               IndexType end = begin + segmentSize;
               IndexType localIdx = 0;
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, localIdx++, j );
            }
            else {
               IndexType localIdx = 0;
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + chunksInSlice * chunkSize;
                  for( IndexType j = begin; j < end; j += chunksInSlice ) {
                     function( segmentIdx, localIdx++, j );
                  }
               }
            }
         };
         Algorithms::parallelFor< DeviceType >( begin, end, work );
      }
      else {  // argumentCount< Function >() == 2
         auto work = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
         {
            if( ! condition( segmentIdx ) )
               return;
            const IndexType sliceIdx = segmentToSliceMapping[ segmentIdx ];

            IndexType firstChunkOfSegment( 0 );
            if( segmentIdx != slices[ sliceIdx ].firstSegment ) {
               firstChunkOfSegment = segmentToChunkMapping[ segmentIdx - 1 ];
            }

            const IndexType lastChunkOfSegment = segmentToChunkMapping[ segmentIdx ];
            const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
            const IndexType sliceOffset = slices[ sliceIdx ].pointer;
            const IndexType chunkSize = slices[ sliceIdx ].chunkSize;

            const IndexType segmentSize = segmentChunksCount * chunkSize;
            if( Organization == RowMajorOrder ) {
               IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
               IndexType end = begin + segmentSize;
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, j );
            }
            else {
               IndexType localIdx = 0;
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + chunksInSlice * chunkSize;
                  for( IndexType j = begin; j < end; j += chunksInSlice ) {
                     function( segmentIdx, j );
                  }
               }
            }
         };
         Algorithms::parallelFor< DeviceType >( begin, end, work );
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ViewType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  const LaunchConfiguration& launchConfig,
                  Condition condition,
                  Function function )
   {
      forElementsIf( segments.getConstView(),
                     begin,
                     end,
                     launchConfig,
                     std::forward< Condition >( condition ),
                     std::forward< Function >( function ) );
   }
};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
struct SegmentsOperations< Segments::ChunkedEllpack< Device, Index, IndexAllocator, Organization > >
{
   // TODO: Rename detail::ChunkedEllpack and fix this
   using SegmentsType = Segments::ChunkedEllpack< Device, Index, IndexAllocator, Organization >;
   using ViewType = typename SegmentsType::ViewType;
   using ConstViewType = typename SegmentsType::ViewType;
   using DeviceType = Device;
   using IndexType = Index;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const SegmentsType& segments,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      SegmentsOperations< ViewType >::forElements(
         segments.getConstView(), begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const SegmentsType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      SegmentsOperations< ViewType >::forElements(
         segments.getConstView(), segmentIndexes, begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const SegmentsType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  const LaunchConfiguration& launchConfig,
                  Condition&& condition,
                  Function&& function )
   {
      SegmentsOperations< ViewType >::forElementsIf( segments.getConstView(),
                                                     begin,
                                                     end,
                                                     launchConfig,
                                                     std::forward< Condition >( condition ),
                                                     std::forward< Function >( function ) );
   }
};
}  //namespace TNL::Algorithms::Segments::detail
