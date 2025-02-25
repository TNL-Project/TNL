// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "ChunkedEllpackBase.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
ChunkedEllpackBase< Device, Index, Organization >::bind( IndexType size,
                                                         IndexType storageSize,
                                                         IndexType numberOfSlices,
                                                         IndexType chunksInSlice,
                                                         IndexType desiredChunkSize,
                                                         OffsetsView segmentToChunkMapping,
                                                         OffsetsView segmentToSliceMapping,
                                                         OffsetsView chunksToSegmentsMapping,
                                                         OffsetsView segmentPointers,
                                                         SliceInfoContainerView slices )
{
   this->size = size;
   this->storageSize = storageSize;
   this->numberOfSlices = numberOfSlices;
   this->chunksInSlice = chunksInSlice;
   this->desiredChunkSize = desiredChunkSize;
   this->segmentToChunkMapping.bind( std::move( segmentToChunkMapping ) );
   this->segmentToSliceMapping.bind( std::move( segmentToSliceMapping ) );
   this->chunksToSegmentsMapping.bind( std::move( chunksToSegmentsMapping ) );
   this->segmentPointers.bind( std::move( segmentPointers ) );
   this->slices.bind( std::move( slices ) );
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
ChunkedEllpackBase< Device, Index, Organization >::ChunkedEllpackBase( IndexType size,
                                                                       IndexType storageSize,
                                                                       IndexType numberOfSlices,
                                                                       IndexType chunksInSlice,
                                                                       IndexType desiredChunkSize,
                                                                       OffsetsView segmentToChunkMapping,
                                                                       OffsetsView segmentToSliceMapping,
                                                                       OffsetsView chunksToSegmentsMapping,
                                                                       OffsetsView segmentPointers,
                                                                       SliceInfoContainerView slices )
: size( size ),
  storageSize( storageSize ),
  numberOfSlices( numberOfSlices ),
  chunksInSlice( chunksInSlice ),
  desiredChunkSize( desiredChunkSize ),
  segmentToChunkMapping( std::move( segmentToChunkMapping ) ),
  segmentToSliceMapping( std::move( segmentToSliceMapping ) ),
  chunksToSegmentsMapping( std::move( chunksToSegmentsMapping ) ),
  segmentPointers( std::move( segmentPointers ) ),
  slices( std::move( slices ) )
{}

template< typename Device, typename Index, ElementsOrganization Organization >
std::string
ChunkedEllpackBase< Device, Index, Organization >::getSerializationType()
{
   return "ChunkedEllpack< " + TNL::getSerializationType< IndexType >() + ", " + TNL::getSerializationType( Organization )
        + " >";
}

template< typename Device, typename Index, ElementsOrganization Organization >
std::string
ChunkedEllpackBase< Device, Index, Organization >::getSegmentsType()
{
   return "ChunkedEllpack";
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSegmentsCount() const -> IndexType
{
   return this->segmentToChunkMapping.getSize();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSegmentSize( IndexType segmentIdx ) const -> IndexType
{
   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentSizeDirect(
         segmentToSliceMapping, slices, segmentToChunkMapping, segmentIdx );
#else
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentSize(
         segmentToSliceMapping, slices, segmentToChunkMapping, segmentIdx );
#endif
   }
   else {
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentSizeDirect(
         segmentToSliceMapping, slices, segmentToChunkMapping, segmentIdx );
   }
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSize() const -> IndexType
{
   return this->size;
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getGlobalIndex( IndexType segmentIdx, IndexType localIdx ) const -> IndexType
{
   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getGlobalIndexDirect(
         segmentToSliceMapping, slices, segmentToChunkMapping, chunksInSlice, segmentIdx, localIdx );
#else
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getGlobalIndex(
         segmentToSliceMapping, slices, segmentToChunkMapping, chunksInSlice, segmentIdx, localIdx );
#endif
   }
   else {
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getGlobalIndexDirect(
         segmentToSliceMapping, slices, segmentToChunkMapping, chunksInSlice, segmentIdx, localIdx );
   }
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSegmentView( IndexType segmentIdx ) const -> SegmentViewType
{
   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentViewDirect(
         segmentToSliceMapping, slices, segmentToChunkMapping, chunksInSlice, segmentIdx );
#else
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentView(
         segmentToSliceMapping, slices, segmentToChunkMapping, chunksInSlice, segmentIdx );
#endif
   }
   else {
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentViewDirect(
         segmentToSliceMapping, slices, segmentToChunkMapping, chunksInSlice, segmentIdx );
   }
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSegmentToChunkMappingView() -> OffsetsView
{
   return segmentToChunkMapping.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSegmentToChunkMappingView() const -> ConstOffsetsView
{
   return segmentToChunkMapping.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSegmentToSliceMappingView() -> OffsetsView
{
   return segmentToSliceMapping.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSegmentToSliceMappingView() const -> ConstOffsetsView
{
   return segmentToSliceMapping.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getChunksToSegmentsMappingView() -> OffsetsView
{
   return chunksToSegmentsMapping.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getChunksToSegmentsMappingView() const -> ConstOffsetsView
{
   return chunksToSegmentsMapping.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSegmentPointersView() -> OffsetsView
{
   return segmentPointers.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSegmentPointersView() const -> ConstOffsetsView
{
   return segmentPointers.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSlicesView() -> SliceInfoContainerView
{
   return slices.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSlicesView() const -> ConstSliceInfoContainerView
{
   return slices.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getNumberOfSlices() const -> IndexType
{
   return numberOfSlices;
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getChunksInSlice() const -> IndexType
{
   return chunksInSlice;
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getDesiredChunkSize() const -> IndexType
{
   return desiredChunkSize;
}

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
ChunkedEllpackBase< Device, Index, Organization >::forElements( IndexType begin, IndexType end, Function&& function ) const
{
   const IndexType chunksInSlice = this->chunksInSlice;
   auto segmentToChunkMapping = this->segmentToChunkMapping;
   auto segmentToSliceMapping = this->segmentToSliceMapping;
   auto slices = this->slices;
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

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
ChunkedEllpackBase< Device, Index, Organization >::forAllElements( Function&& function ) const
{
   this->forElements( 0, this->getSegmentsCount(), function );
}

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Array, typename Function >
void
ChunkedEllpackBase< Device, Index, Organization >::forElements( const Array& segmentIndexes,
                                                                Index begin,
                                                                Index end,
                                                                Function function ) const
{
   const IndexType chunksInSlice = this->chunksInSlice;
   auto segmentToChunkMapping = this->segmentToChunkMapping;
   auto segmentToSliceMapping = this->segmentToSliceMapping;
   auto slices = this->slices;
   auto segmentIndexesView = segmentIndexes.getConstView();
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

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Array, typename Function >
void
ChunkedEllpackBase< Device, Index, Organization >::forElements( const Array& segmentIndexes, Function function ) const
{
   this->forElements( segmentIndexes, 0, segmentIndexes.getSize(), function );
}

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Condition, typename Function >
void
ChunkedEllpackBase< Device, Index, Organization >::forElementsIf( IndexType begin,
                                                                  IndexType end,
                                                                  Condition condition,
                                                                  Function function ) const
{
   const IndexType chunksInSlice = this->chunksInSlice;
   auto segmentToChunkMapping = this->segmentToChunkMapping;
   auto segmentToSliceMapping = this->segmentToSliceMapping;
   auto slices = this->slices;
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

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Condition, typename Function >
void
ChunkedEllpackBase< Device, Index, Organization >::forAllElementsIf( Condition condition, Function function ) const
{
   this->forElementsIf( 0, this->getSegmentsCount(), condition, function );
}

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
ChunkedEllpackBase< Device, Index, Organization >::forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   const auto& self = *this;
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = self.getSegmentView( segmentIdx );
      function( segment );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
ChunkedEllpackBase< Device, Index, Organization >::forAllSegments( Function&& function ) const
{
   this->forSegments( 0, this->getSegmentsCount(), function );
}

template< typename Device, typename Index, ElementsOrganization Organization >
void
ChunkedEllpackBase< Device, Index, Organization >::printStructure( std::ostream& str ) const
{
   str << "Segments count: " << this->getSegmentsCount() << std::endl << "Slices: " << this->getNumberOfSlices() << std::endl;
   for( IndexType i = 0; i < this->getNumberOfSlices(); i++ )
      str << "   Slice " << i << " : size = " << this->slices.getElement( i ).size
          << " chunkSize = " << this->slices.getElement( i ).chunkSize
          << " firstSegment = " << this->slices.getElement( i ).firstSegment
          << " pointer = " << this->slices.getElement( i ).pointer << std::endl;
   for( IndexType i = 0; i < this->getSegmentsCount(); i++ )
      str << "Segment " << i << " : slice = " << this->segmentToSliceMapping.getElement( i )
          << " chunk = " << this->segmentToChunkMapping.getElement( i ) << std::endl;
}

}  // namespace TNL::Algorithms::Segments
