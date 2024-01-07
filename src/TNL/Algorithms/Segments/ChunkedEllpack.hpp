// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/scan.h>

#include "ChunkedEllpack.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::ChunkedEllpack( const ChunkedEllpack& segments )
: rowToChunkMapping( segments.rowToChunkMapping ), rowToSliceMapping( segments.rowToSliceMapping ),
  chunksToSegmentsMapping( segments.chunksToSegmentsMapping ), rowPointers( segments.rowPointers ), slices( segments.slices )
{
   // update the base
   Base::bind( segments.getSize(),
               segments.getStorageSize(),
               segments.getNumberOfSlices(),
               segments.getChunksInSlice(),
               segments.getDesiredChunkSize(),
               this->rowToChunkMapping.getView(),
               this->rowToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->rowPointers.getView(),
               this->slices.getView() );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
template< typename SizesContainer >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::ChunkedEllpack( const SizesContainer& segmentsSizes )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
template< typename ListIndex >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::ChunkedEllpack(
   const std::initializer_list< ListIndex >& segmentsSizes )
{
   this->setSegmentsSizes( OffsetsContainer( segmentsSizes ) );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >&
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::operator=( const ChunkedEllpack& segments )
{
   this->rowToChunkMapping = segments.rowToChunkMapping;
   this->rowToSliceMapping = segments.rowToSliceMapping;
   this->rowPointers = segments.rowPointers;
   this->chunksToSegmentsMapping = segments.chunksToSegmentsMapping;
   this->slices = segments.slices;
   // update the base
   Base::bind( segments.getSize(),
               segments.getStorageSize(),
               segments.getNumberOfSlices(),
               segments.getChunksInSlice(),
               segments.getDesiredChunkSize(),
               this->rowToChunkMapping.getView(),
               this->rowToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->rowPointers.getView(),
               this->slices.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >&
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::operator=( ChunkedEllpack&& segments ) noexcept( false )
{
   this->rowToChunkMapping = std::move( segments.rowToChunkMapping );
   this->rowToSliceMapping = std::move( segments.rowToSliceMapping );
   this->rowPointers = std::move( segments.rowPointers );
   this->chunksToSegmentsMapping = std::move( segments.chunksToSegmentsMapping );
   this->slices = std::move( segments.slices );
   // update the base
   Base::bind( segments.getSize(),
               segments.getStorageSize(),
               segments.getNumberOfSlices(),
               segments.getChunksInSlice(),
               segments.getDesiredChunkSize(),
               this->rowToChunkMapping.getView(),
               this->rowToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->rowPointers.getView(),
               this->slices.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >&
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::operator=(
   const ChunkedEllpack< Device_, Index_, IndexAllocator_, Organization_ >& segments )
{
   this->rowToChunkMapping = segments.getRowToChunkMappingView();
   this->rowToSliceMapping = segments.getRowToSliceMappingView();
   this->rowPointers = segments.getRowPointersView();
   this->chunksToSegmentsMapping = segments.getChunksToSegmentsMappingView();
   this->slices = segments.getSlicesView();
   // update the base
   Base::bind( segments.getSize(),
               segments.getStorageSize(),
               segments.getNumberOfSlices(),
               segments.getChunksInSlice(),
               segments.getDesiredChunkSize(),
               this->rowToChunkMapping.getView(),
               this->rowToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->rowPointers.getView(),
               this->slices.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
typename ChunkedEllpack< Device, Index, IndexAllocator, Organization >::ViewType
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::getView()
{
   return { this->getSize(),
            this->getStorageSize(),
            this->getNumberOfSlices(),
            this->getChunksInSlice(),
            this->getDesiredChunkSize(),
            this->getRowToChunkMappingView(),
            this->getRowToSliceMappingView(),
            this->getChunksToSegmentsMappingView(),
            this->getRowPointersView(),
            this->getSlicesView() };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
auto
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::getConstView() const -> ConstViewType
{
   return { this->getSize(),
            this->getStorageSize(),
            this->getNumberOfSlices(),
            this->getChunksInSlice(),
            this->getDesiredChunkSize(),
            this->getRowToChunkMappingView(),
            this->getRowToSliceMappingView(),
            this->getChunksToSegmentsMappingView(),
            this->getRowPointersView(),
            this->getSlicesView() };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
template< typename SizesContainer >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::setSegmentsSizes( const SizesContainer& segmentsSizes )
{
   if constexpr( std::is_same< Device, Devices::Host >::value ) {
      this->size = sum( segmentsSizes );
      const Index segmentsCount = segmentsSizes.getSize();
      this->slices.setSize( segmentsCount );
      this->rowToChunkMapping.setSize( segmentsCount );
      this->rowToSliceMapping.setSize( segmentsCount );
      this->rowPointers.setSize( segmentsCount + 1 );

      this->resolveSliceSizes( segmentsSizes );
      this->rowPointers.setElement( 0, 0 );
      this->storageSize = 0;
      for( Index sliceIndex = 0; sliceIndex < this->numberOfSlices; sliceIndex++ )
         this->setSlice( segmentsSizes, sliceIndex, this->storageSize );
      inplaceInclusiveScan( this->rowPointers );

      Index chunksCount = this->numberOfSlices * this->chunksInSlice;
      this->chunksToSegmentsMapping.setSize( chunksCount );
      Index chunkIdx = 0;
      for( Index segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         const Index& sliceIdx = rowToSliceMapping[ segmentIdx ];
         Index firstChunkOfSegment = 0;
         if( segmentIdx != slices[ sliceIdx ].firstSegment )
            firstChunkOfSegment = rowToChunkMapping[ segmentIdx - 1 ];

         const Index lastChunkOfSegment = rowToChunkMapping[ segmentIdx ];
         const Index segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         for( Index i = 0; i < segmentChunksCount; i++ )
            this->chunksToSegmentsMapping[ chunkIdx++ ] = segmentIdx;
      }

      // update the base
      Base::bind( this->size,
                  this->storageSize,
                  this->numberOfSlices,
                  this->chunksInSlice,
                  this->desiredChunkSize,
                  this->rowToChunkMapping.getView(),
                  this->rowToSliceMapping.getView(),
                  this->chunksToSegmentsMapping.getView(),
                  this->rowPointers.getView(),
                  this->slices.getView() );
   }
   else {
      ChunkedEllpack< Devices::Host,
                      Index,
                      typename Allocators::Default< Devices::Host >::template Allocator< Index >,
                      Organization >
         hostSegments;
      Containers::Vector< Index, Devices::Host, Index > hostSegmentsSizes;
      hostSegmentsSizes = segmentsSizes;
      hostSegments.setSegmentsSizes( hostSegmentsSizes );
      *this = hostSegments;
   }
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::reset()
{
   this->rowToChunkMapping.reset();
   this->rowToSliceMapping.reset();
   this->chunksToSegmentsMapping.reset();
   this->rowPointers.reset();
   this->slices.reset();

   // update the base
   Base::bind( 0,
               0,
               0,
               this->getChunksInSlice(),
               this->getDesiredChunkSize(),
               this->rowToChunkMapping.getView(),
               this->rowToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->rowPointers.getView(),
               this->slices.getView() );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->numberOfSlices );
   file.save( &this->chunksInSlice );
   file.save( &this->desiredChunkSize );
   file << this->rowToChunkMapping << this->rowToSliceMapping << this->chunksToSegmentsMapping << this->rowPointers
        << this->slices;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file.load( &this->numberOfSlices );
   file.load( &this->chunksInSlice );
   file.load( &this->desiredChunkSize );
   file >> this->rowToChunkMapping >> this->rowToSliceMapping >> this->chunksToSegmentsMapping >> this->rowPointers
      >> this->slices;

   // update the base
   Base::bind( this->size,
               this->storageSize,
               this->numberOfSlices,
               this->chunksInSlice,
               this->desiredChunkSize,
               this->rowToChunkMapping.getView(),
               this->rowToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->rowPointers.getView(),
               this->slices.getView() );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
template< typename SegmentsSizes >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::resolveSliceSizes( SegmentsSizes& segmentsSizes )
{
   // Iterate over rows and allocate slices so that each slice has
   // approximately the same number of allocated elements
   const Index desiredElementsInSlice = this->chunksInSlice * this->desiredChunkSize;

   Index segmentIdx = 0;
   Index sliceSize = 0;
   Index allocatedElementsInSlice = 0;
   this->numberOfSlices = 0;
   while( segmentIdx < segmentsSizes.getSize() ) {
      // Add one row to the current slice until we reach the desired
      // number of elements in a slice.
      allocatedElementsInSlice += segmentsSizes[ segmentIdx ];
      sliceSize++;
      segmentIdx++;
      if( allocatedElementsInSlice < desiredElementsInSlice )
         if( segmentIdx < segmentsSizes.getSize() && sliceSize < this->chunksInSlice )
            continue;
      TNL_ASSERT_GT( sliceSize, 0, "" );
      this->slices[ this->numberOfSlices ].size = sliceSize;
      this->slices[ this->numberOfSlices ].firstSegment = segmentIdx - sliceSize;
      this->slices[ this->numberOfSlices ].pointer = allocatedElementsInSlice;  // this is only temporary
      sliceSize = 0;
      this->numberOfSlices++;
      allocatedElementsInSlice = 0;
   }
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
template< typename SizesContainer >
bool
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::setSlice( SizesContainer& segmentsSizes,
                                                                         Index sliceIndex,
                                                                         Index& elementsToAllocation )
{
   /* Now, compute the number of chunks per each row. Each row gets one chunk
    * by default. Then each row will get additional chunks with respect to the
    * number of the elements in the row. If there are some free chunks left,
    * repeat it again.
    */
   const Index sliceSize = this->slices[ sliceIndex ].size;
   const Index sliceBegin = this->slices[ sliceIndex ].firstSegment;
   const Index allocatedElementsInSlice = this->slices[ sliceIndex ].pointer;
   const Index sliceEnd = sliceBegin + sliceSize;

   Index freeChunks = this->chunksInSlice - sliceSize;
   for( Index i = sliceBegin; i < sliceEnd; i++ )
      this->rowToChunkMapping.setElement( i, 1 );

   int totalAddedChunks = 0;
   int maxRowLength( segmentsSizes[ sliceBegin ] );
   for( Index i = sliceBegin; i < sliceEnd; i++ ) {
      double rowRatio = 0.0;
      if( allocatedElementsInSlice != 0 )
         rowRatio = (double) segmentsSizes[ i ] / (double) allocatedElementsInSlice;
      const Index addedChunks = freeChunks * rowRatio;
      totalAddedChunks += addedChunks;
      this->rowToChunkMapping[ i ] += addedChunks;
      if( maxRowLength < segmentsSizes[ i ] )
         maxRowLength = segmentsSizes[ i ];
   }
   TNL_ASSERT_GE( freeChunks, totalAddedChunks, "" );
   freeChunks -= totalAddedChunks;
   while( freeChunks )
      for( Index i = sliceBegin; i < sliceEnd && freeChunks; i++ )
         if( segmentsSizes[ i ] == maxRowLength ) {
            this->rowToChunkMapping[ i ]++;
            freeChunks--;
         }

   // Compute the chunk size
   Index maxChunkInSlice = 0;
   for( Index i = sliceBegin; i < sliceEnd; i++ ) {
      TNL_ASSERT_NE( this->rowToChunkMapping[ i ], 0, "" );
      maxChunkInSlice = TNL::max( maxChunkInSlice, roundUpDivision( segmentsSizes[ i ], this->rowToChunkMapping[ i ] ) );
   }

   // Set up the slice info.
   this->slices[ sliceIndex ].chunkSize = maxChunkInSlice;
   this->slices[ sliceIndex ].pointer = elementsToAllocation;
   elementsToAllocation += this->chunksInSlice * maxChunkInSlice;

   for( Index i = sliceBegin; i < sliceEnd; i++ )
      this->rowToSliceMapping[ i ] = sliceIndex;

   for( Index i = sliceBegin; i < sliceEnd; i++ ) {
      this->rowPointers[ i + 1 ] = maxChunkInSlice * rowToChunkMapping[ i ];
      TNL_ASSERT_GE( this->rowPointers[ i ], 0, "" );
      TNL_ASSERT_GE( this->rowPointers[ i + 1 ], 0, "" );
   }

   // Finish the row to chunk mapping by computing the prefix sum.
   for( Index j = sliceBegin + 1; j < sliceEnd; j++ )
      rowToChunkMapping[ j ] += rowToChunkMapping[ j - 1 ];
   return true;
}

}  // namespace TNL::Algorithms::Segments
