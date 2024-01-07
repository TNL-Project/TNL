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
                                                         OffsetsView rowToChunkMapping,
                                                         OffsetsView rowToSliceMapping,
                                                         OffsetsView chunksToSegmentsMapping,
                                                         OffsetsView rowPointers,
                                                         SliceInfoContainerView slices )
{
   this->size = size;
   this->storageSize = storageSize;
   this->numberOfSlices = numberOfSlices;
   this->chunksInSlice = chunksInSlice;
   this->desiredChunkSize = desiredChunkSize;
   this->rowToChunkMapping.bind( std::move( rowToChunkMapping ) );
   this->rowToSliceMapping.bind( std::move( rowToSliceMapping ) );
   this->chunksToSegmentsMapping.bind( std::move( chunksToSegmentsMapping ) );
   this->rowPointers.bind( std::move( rowPointers ) );
   this->slices.bind( std::move( slices ) );
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
ChunkedEllpackBase< Device, Index, Organization >::ChunkedEllpackBase( IndexType size,
                                                                       IndexType storageSize,
                                                                       IndexType numberOfSlices,
                                                                       IndexType chunksInSlice,
                                                                       IndexType desiredChunkSize,
                                                                       OffsetsView rowToChunkMapping,
                                                                       OffsetsView rowToSliceMapping,
                                                                       OffsetsView chunksToSegmentsMapping,
                                                                       OffsetsView rowPointers,
                                                                       SliceInfoContainerView slices )
: size( size ), storageSize( storageSize ), numberOfSlices( numberOfSlices ), chunksInSlice( chunksInSlice ),
  desiredChunkSize( desiredChunkSize ), rowToChunkMapping( std::move( rowToChunkMapping ) ),
  rowToSliceMapping( std::move( rowToSliceMapping ) ), chunksToSegmentsMapping( std::move( chunksToSegmentsMapping ) ),
  rowPointers( std::move( rowPointers ) ), slices( std::move( slices ) )
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
   return this->size;
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSegmentSize( IndexType segmentIdx ) const -> IndexType
{
   if( std::is_same< DeviceType, Devices::Host >::value )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentSizeDirect(
         rowToSliceMapping, slices, rowToChunkMapping, segmentIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentSizeDirect(
         rowToSliceMapping, slices, rowToChunkMapping, segmentIdx );
#else
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentSize(
         rowToSliceMapping, slices, rowToChunkMapping, segmentIdx );
#endif
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
   if( std::is_same< DeviceType, Devices::Host >::value )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getGlobalIndexDirect(
         rowToSliceMapping, slices, rowToChunkMapping, chunksInSlice, segmentIdx, localIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getGlobalIndexDirect(
         rowToSliceMapping, slices, rowToChunkMapping, chunksInSlice, segmentIdx, localIdx );
#else
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getGlobalIndex(
         rowToSliceMapping, slices, rowToChunkMapping, chunksInSlice, segmentIdx, localIdx );
#endif
   }
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getSegmentView( IndexType segmentIdx ) const -> SegmentViewType
{
   if( std::is_same< DeviceType, Devices::Host >::value )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentViewDirect(
         rowToSliceMapping, slices, rowToChunkMapping, chunksInSlice, segmentIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentViewDirect(
         rowToSliceMapping, slices, rowToChunkMapping, chunksInSlice, segmentIdx );
#else
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentView(
         rowToSliceMapping, slices, rowToChunkMapping, chunksInSlice, segmentIdx );
#endif
   }
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getRowToChunkMappingView() -> OffsetsView
{
   return rowToChunkMapping.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getRowToChunkMappingView() const -> ConstOffsetsView
{
   return rowToChunkMapping.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getRowToSliceMappingView() -> OffsetsView
{
   return rowToSliceMapping.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getRowToSliceMappingView() const -> ConstOffsetsView
{
   return rowToSliceMapping.getConstView();
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
ChunkedEllpackBase< Device, Index, Organization >::getRowPointersView() -> OffsetsView
{
   return rowPointers.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackBase< Device, Index, Organization >::getRowPointersView() const -> ConstOffsetsView
{
   return rowPointers.getConstView();
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
   auto rowToChunkMapping = this->rowToChunkMapping;
   auto rowToSliceMapping = this->rowToSliceMapping;
   auto slices = this->slices;
   auto work = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      const IndexType sliceIdx = rowToSliceMapping[ segmentIdx ];

      IndexType firstChunkOfSegment( 0 );
      if( segmentIdx != slices[ sliceIdx ].firstSegment ) {
         firstChunkOfSegment = rowToChunkMapping[ segmentIdx - 1 ];
      }

      const IndexType lastChunkOfSegment = rowToChunkMapping[ segmentIdx ];
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
   str << "Segments count: " << this->getSize() << std::endl << "Slices: " << this->getNumberOfSlices() << std::endl;
   for( IndexType i = 0; i < this->getNumberOfSlices(); i++ )
      str << "   Slice " << i << " : size = " << this->slices.getElement( i ).size
          << " chunkSize = " << this->slices.getElement( i ).chunkSize
          << " firstSegment = " << this->slices.getElement( i ).firstSegment
          << " pointer = " << this->slices.getElement( i ).pointer << std::endl;
   for( IndexType i = 0; i < this->getSize(); i++ )
      str << "Segment " << i << " : slice = " << this->rowToSliceMapping.getElement( i )
          << " chunk = " << this->rowToChunkMapping.getElement( i ) << std::endl;
}

}  // namespace TNL::Algorithms::Segments
