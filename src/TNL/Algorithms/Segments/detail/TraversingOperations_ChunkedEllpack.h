// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ChunkedEllpackView.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include "TraversingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization >
struct TraversingOperations< ChunkedEllpackView< Device, Index, Organization > >
: public TraversingOperationsBase< ChunkedEllpackView< Device, Index, Organization > >
{
   using ViewType = Segments::ChunkedEllpackView< Device, Index, Organization >;
   using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;
   using ConstOffsetsView = typename ViewType::ConstOffsetsView;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                const LaunchConfiguration& launchConfig )
   {
      const IndexType chunksInSlice = segments.getChunksInSlice();
      auto segmentToChunkMapping = segments.getSegmentToChunkMappingView();
      auto segmentToSliceMapping = segments.getSegmentToSliceMappingView();
      auto slices = segments.getSlicesView();
      auto work = [ chunksInSlice, segmentToChunkMapping, segmentToSliceMapping, slices, function ] __cuda_callable__(
                     IndexType segmentIdx ) mutable
      {
         (void) chunksInSlice;  // To suppress unused variable warning
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
         if constexpr( Organization == RowMajorOrder ) {
            IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
            IndexType end = begin + segmentSize;
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx = 0;
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, localIdx++, j );
            }
            else {
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, j );
            }
         }
         else {
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx = 0;
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + chunksInSlice * chunkSize;
                  for( IndexType j = begin; j < end; j += chunksInSlice ) {
                     function( segmentIdx, localIdx++, j );
                  }
               }
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
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, work );
   }

   template< typename Array, typename Function >
   static void
   forElements( const ConstViewType& segments,
                const Array& segmentIndexes,
                Function&& function,
                const LaunchConfiguration& launchConfig )
   {
      const IndexType chunksInSlice = segments.getChunksInSlice();
      auto segmentToChunkMapping = segments.getSegmentToChunkMappingView();
      auto segmentToSliceMapping = segments.getSegmentToSliceMappingView();
      auto slices = segments.getSlicesView();
      auto segmentIndexesView = segmentIndexes.getConstView();
      auto work = [ chunksInSlice,
                    segmentToChunkMapping,
                    segmentToSliceMapping,
                    slices,
                    segmentIndexesView,
                    function ] __cuda_callable__( IndexType idx ) mutable
      {
         (void) chunksInSlice;  // To suppress unused variable warning
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
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx = 0;
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, localIdx++, j );
            }
            else {
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, j );
            }
         }
         else {
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx = 0;
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + chunksInSlice * chunkSize;
                  for( IndexType j = begin; j < end; j += chunksInSlice ) {
                     function( segmentIdx, localIdx++, j );
                  }
               }
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
         }
      };
      Algorithms::parallelFor< DeviceType >( 0, segmentIndexes.getSize(), work );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ConstViewType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Function&& function,
                  const LaunchConfiguration& launchConfig )
   {
      const IndexType chunksInSlice = segments.getChunksInSlice();
      auto segmentToChunkMapping = segments.getSegmentToChunkMappingView();
      auto segmentToSliceMapping = segments.getSegmentToSliceMappingView();
      auto slices = segments.getSlicesView();
      auto work =
         [ chunksInSlice, segmentToChunkMapping, segmentToSliceMapping, slices, condition, function ] __cuda_callable__(
            IndexType segmentIdx ) mutable
      {
         (void) chunksInSlice;  // To suppress unused variable warning
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
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx = 0;
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, localIdx++, j );
            }
            else {
               for( IndexType j = begin; j < end; j++ )
                  function( segmentIdx, j );
            }
         }
         else {
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx = 0;
               for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
                  IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
                  IndexType end = begin + chunksInSlice * chunkSize;
                  for( IndexType j = begin; j < end; j += chunksInSlice ) {
                     function( segmentIdx, localIdx++, j );
                  }
               }
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
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, work );
   }
};

}  //namespace TNL::Algorithms::Segments::detail
