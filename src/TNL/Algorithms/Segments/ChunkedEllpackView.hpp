// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "ChunkedEllpackView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
ChunkedEllpackView< Device, Index, Organization >::ChunkedEllpackView( const IndexType size,
                                                                       const IndexType storageSize,
                                                                       const IndexType chunksInSlice,
                                                                       const IndexType desiredChunkSize,
                                                                       const OffsetsView& rowToChunkMapping,
                                                                       const OffsetsView& rowToSliceMapping,
                                                                       const OffsetsView& chunksToSegmentsMapping,
                                                                       const OffsetsView& rowPointers,
                                                                       const ChunkedEllpackSliceInfoContainerView& slices,
                                                                       const IndexType numberOfSlices )
: size( size ), storageSize( storageSize ), numberOfSlices( numberOfSlices ), chunksInSlice( chunksInSlice ),
  desiredChunkSize( desiredChunkSize ), rowToSliceMapping( rowToSliceMapping ), rowToChunkMapping( rowToChunkMapping ),
  chunksToSegmentsMapping( chunksToSegmentsMapping ), rowPointers( rowPointers ), slices( slices )
{}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
ChunkedEllpackView< Device, Index, Organization >::ChunkedEllpackView( const IndexType size,
                                                                       const IndexType storageSize,
                                                                       const IndexType chunksInSlice,
                                                                       const IndexType desiredChunkSize,
                                                                       const OffsetsView&& rowToChunkMapping,
                                                                       const OffsetsView&& rowToSliceMapping,
                                                                       const OffsetsView&& chunksToSegmentsMapping,
                                                                       const OffsetsView&& rowPointers,
                                                                       const ChunkedEllpackSliceInfoContainerView&& slices,
                                                                       const IndexType numberOfSlices )
: size( size ), storageSize( storageSize ), numberOfSlices( numberOfSlices ), chunksInSlice( chunksInSlice ),
  desiredChunkSize( desiredChunkSize ), rowToSliceMapping( std::move( rowToSliceMapping ) ),
  rowToChunkMapping( std::move( rowToChunkMapping ) ), chunksToSegmentsMapping( std::move( chunksToSegmentsMapping ) ),
  rowPointers( std::move( rowPointers ) ), slices( std::move( slices ) )
{}

template< typename Device, typename Index, ElementsOrganization Organization >
std::string
ChunkedEllpackView< Device, Index, Organization >::getSerializationType()
{
   return "ChunkedEllpack< " + TNL::getSerializationType< IndexType >() + ", " + TNL::getSerializationType( Organization )
        + " >";
}

template< typename Device, typename Index, ElementsOrganization Organization >
String
ChunkedEllpackView< Device, Index, Organization >::getSegmentsType()
{
   return "ChunkedEllpack";
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
typename ChunkedEllpackView< Device, Index, Organization >::ViewType
ChunkedEllpackView< Device, Index, Organization >::getView()
{
   return { size,
            storageSize,
            chunksInSlice,
            desiredChunkSize,
            rowToChunkMapping.getView(),
            rowToSliceMapping.getView(),
            chunksToSegmentsMapping.getView(),
            rowPointers.getView(),
            slices.getView(),
            numberOfSlices };
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::getConstView() const -> ConstViewType
{
   return { size,
            storageSize,
            chunksInSlice,
            desiredChunkSize,
            rowToChunkMapping.getConstView(),
            rowToSliceMapping.getConstView(),
            chunksToSegmentsMapping.getConstView(),
            rowPointers.getConstView(),
            slices.getConstView(),
            numberOfSlices };
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::getSegmentsCount() const -> IndexType
{
   return this->size;
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   if( std::is_same< DeviceType, Devices::Host >::value )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentSizeDirect(
         rowToSliceMapping, slices, rowToChunkMapping, segmentIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
#ifdef __CUDA_ARCH__
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
ChunkedEllpackView< Device, Index, Organization >::getSize() const -> IndexType
{
   return this->size;
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::getGlobalIndex( const Index segmentIdx, const Index localIdx ) const
   -> IndexType
{
   if( std::is_same< DeviceType, Devices::Host >::value )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getGlobalIndexDirect(
         rowToSliceMapping, slices, rowToChunkMapping, chunksInSlice, segmentIdx, localIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
#ifdef __CUDA_ARCH__
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
ChunkedEllpackView< Device, Index, Organization >::getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   if( std::is_same< DeviceType, Devices::Host >::value )
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentViewDirect(
         rowToSliceMapping, slices, rowToChunkMapping, chunksInSlice, segmentIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
#ifdef __CUDA_ARCH__
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
ChunkedEllpackView< Device, Index, Organization >::getSlicesView() const -> ChunkedEllpackSliceInfoConstView
{
   return slices.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::getRowToChunkMappingView() const -> ConstOffsetsView
{
   return rowToChunkMapping.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::getRowToSliceMappingView() const -> ConstOffsetsView
{
   return rowToSliceMapping.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::getChunksToSegmentsMappingView() const -> ConstOffsetsView
{
   return chunksToSegmentsMapping.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::getChunksInSlice() const -> IndexType
{
   return chunksInSlice;
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::getNumberOfSlices() const -> IndexType
{
   return numberOfSlices;
}

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
ChunkedEllpackView< Device, Index, Organization >::forElements( IndexType first, IndexType last, Function&& f ) const
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
         IndexType localIdx( 0 );
         for( IndexType j = begin; j < end; j++ )
            f( segmentIdx, localIdx++, j );
      }
      else {
         IndexType localIdx( 0 );
         for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
            IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
            IndexType end = begin + chunksInSlice * chunkSize;
            for( IndexType j = begin; j < end; j += chunksInSlice ) {
               f( segmentIdx, localIdx++, j );
            }
         }
      }
   };
   Algorithms::parallelFor< DeviceType >( first, last, work );
}

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
ChunkedEllpackView< Device, Index, Organization >::forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
ChunkedEllpackView< Device, Index, Organization >::forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   auto view = this->getConstView();
   using SVType = decltype( view.getSegmentView( IndexType() ) );
   static_assert( std::is_same< SVType, SegmentViewType >::value, "" );
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = view.getSegmentView( segmentIdx );
      function( segment );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
ChunkedEllpackView< Device, Index, Organization >::forAllSegments( Function&& f ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, ElementsOrganization Organization >
ChunkedEllpackView< Device, Index, Organization >&
ChunkedEllpackView< Device, Index, Organization >::operator=( const ChunkedEllpackView& view )
{
   this->size = view.size;
   this->storageSize = view.storageSize;
   this->chunksInSlice = view.chunksInSlice;
   this->desiredChunkSize = view.desiredChunkSize;
   this->rowToChunkMapping.bind( view.rowToChunkMapping );
   this->chunksToSegmentsMapping.bind( view.chunksToSegmentsMapping );
   this->rowToSliceMapping.bind( view.rowToSliceMapping );
   this->rowPointers.bind( view.rowPointers );
   this->slices.bind( view.slices );
   this->numberOfSlices = view.numberOfSlices;
   return *this;
}

template< typename Device, typename Index, ElementsOrganization Organization >
void
ChunkedEllpackView< Device, Index, Organization >::save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->chunksInSlice );
   file.save( &this->desiredChunkSize );
   file << this->rowToChunkMapping << this->chunksToSegmentsMapping << this->rowToSliceMapping << this->rowPointers
        << this->slices;
   file.save( &this->numberOfSlices );
}

template< typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch >
auto
ChunkedEllpackView< Device, Index, Organization >::print( Fetch&& fetch ) const -> SegmentsPrinter< ChunkedEllpackView, Fetch >
{
   return SegmentsPrinter< ChunkedEllpackView, Fetch >( *this, fetch );
}

template< typename Device, typename Index, ElementsOrganization Organization >
void
ChunkedEllpackView< Device, Index, Organization >::printStructure( std::ostream& str ) const
{
   // const IndexType numberOfSlices = this->getNumberOfSlices();
   str << "Segments count: " << this->getSize() << std::endl << "Slices: " << numberOfSlices << std::endl;
   for( IndexType i = 0; i < numberOfSlices; i++ )
      str << "   Slice " << i << " : size = " << this->slices.getElement( i ).size
          << " chunkSize = " << this->slices.getElement( i ).chunkSize
          << " firstSegment = " << this->slices.getElement( i ).firstSegment
          << " pointer = " << this->slices.getElement( i ).pointer << std::endl;
   for( IndexType i = 0; i < this->getSize(); i++ )
      str << "Segment " << i << " : slice = " << this->rowToSliceMapping.getElement( i )
          << " chunk = " << this->rowToChunkMapping.getElement( i ) << std::endl;
}

}  // namespace TNL::Algorithms::Segments
