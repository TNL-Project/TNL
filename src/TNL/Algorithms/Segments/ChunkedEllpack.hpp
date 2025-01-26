// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/scan.h>

#include "ChunkedEllpack.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::ChunkedEllpack( const ChunkedEllpack& segments )
: segmentToChunkMapping( segments.segmentToChunkMapping ),
  segmentToSliceMapping( segments.segmentToSliceMapping ),
  chunksToSegmentsMapping( segments.chunksToSegmentsMapping ),
  segmentPointers( segments.segmentPointers ),
  slices( segments.slices )
{
   // update the base
   Base::bind( segments.getSize(),
               segments.getStorageSize(),
               segments.getNumberOfSlices(),
               segments.getChunksInSlice(),
               segments.getDesiredChunkSize(),
               this->segmentToChunkMapping.getView(),
               this->segmentToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->segmentPointers.getView(),
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
   this->segmentToChunkMapping = segments.segmentToChunkMapping;
   this->segmentToSliceMapping = segments.segmentToSliceMapping;
   this->segmentPointers = segments.segmentPointers;
   this->chunksToSegmentsMapping = segments.chunksToSegmentsMapping;
   this->slices = segments.slices;
   // update the base
   Base::bind( segments.getSize(),
               segments.getStorageSize(),
               segments.getNumberOfSlices(),
               segments.getChunksInSlice(),
               segments.getDesiredChunkSize(),
               this->segmentToChunkMapping.getView(),
               this->segmentToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->segmentPointers.getView(),
               this->slices.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >&
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::operator=( ChunkedEllpack&& segments ) noexcept( false )
{
   this->segmentToChunkMapping = std::move( segments.segmentToChunkMapping );
   this->segmentToSliceMapping = std::move( segments.segmentToSliceMapping );
   this->segmentPointers = std::move( segments.segmentPointers );
   this->chunksToSegmentsMapping = std::move( segments.chunksToSegmentsMapping );
   this->slices = std::move( segments.slices );
   // update the base
   Base::bind( segments.getSize(),
               segments.getStorageSize(),
               segments.getNumberOfSlices(),
               segments.getChunksInSlice(),
               segments.getDesiredChunkSize(),
               this->segmentToChunkMapping.getView(),
               this->segmentToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->segmentPointers.getView(),
               this->slices.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >&
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::operator=(
   const ChunkedEllpack< Device_, Index_, IndexAllocator_, Organization_ >& segments )
{
   this->segmentToChunkMapping = segments.getSegmentToChunkMappingView();
   this->segmentToSliceMapping = segments.getSegmentToSliceMappingView();
   this->segmentPointers = segments.getSegmentPointersView();
   this->chunksToSegmentsMapping = segments.getChunksToSegmentsMappingView();
   this->slices = segments.getSlicesView();
   // update the base
   Base::bind( segments.getSize(),
               segments.getStorageSize(),
               segments.getNumberOfSlices(),
               segments.getChunksInSlice(),
               segments.getDesiredChunkSize(),
               this->segmentToChunkMapping.getView(),
               this->segmentToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->segmentPointers.getView(),
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
            this->getSegmentToChunkMappingView(),
            this->getSegmentToSliceMappingView(),
            this->getChunksToSegmentsMappingView(),
            this->getSegmentPointersView(),
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
            this->getSegmentToChunkMappingView(),
            this->getSegmentToSliceMappingView(),
            this->getChunksToSegmentsMappingView(),
            this->getSegmentPointersView(),
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
      this->segmentToChunkMapping.setSize( segmentsCount );
      this->segmentToSliceMapping.setSize( segmentsCount );
      this->segmentPointers.setSize( segmentsCount + 1 );

      this->resolveSliceSizes( segmentsSizes );
      this->segmentPointers.setElement( 0, 0 );
      this->storageSize = 0;
      for( Index sliceIndex = 0; sliceIndex < this->numberOfSlices; sliceIndex++ )
         this->setSlice( segmentsSizes, sliceIndex, this->storageSize );
      inplaceInclusiveScan( this->segmentPointers );

      Index chunksCount = this->numberOfSlices * this->chunksInSlice;
      this->chunksToSegmentsMapping.setSize( chunksCount );
      Index chunkIdx = 0;
      for( Index segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         const Index& sliceIdx = segmentToSliceMapping[ segmentIdx ];
         Index firstChunkOfSegment = 0;
         if( segmentIdx != slices[ sliceIdx ].firstSegment )
            firstChunkOfSegment = segmentToChunkMapping[ segmentIdx - 1 ];

         const Index lastChunkOfSegment = segmentToChunkMapping[ segmentIdx ];
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
                  this->segmentToChunkMapping.getView(),
                  this->segmentToSliceMapping.getView(),
                  this->chunksToSegmentsMapping.getView(),
                  this->segmentPointers.getView(),
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
   this->segmentToChunkMapping.reset();
   this->segmentToSliceMapping.reset();
   this->chunksToSegmentsMapping.reset();
   this->segmentPointers.reset();
   this->slices.reset();

   // update the base
   Base::bind( 0,
               0,
               0,
               this->getChunksInSlice(),
               this->getDesiredChunkSize(),
               this->segmentToChunkMapping.getView(),
               this->segmentToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->segmentPointers.getView(),
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
   file << this->segmentToChunkMapping << this->segmentToSliceMapping << this->chunksToSegmentsMapping << this->segmentPointers
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
   file >> this->segmentToChunkMapping >> this->segmentToSliceMapping >> this->chunksToSegmentsMapping >> this->segmentPointers
      >> this->slices;

   // update the base
   Base::bind( this->size,
               this->storageSize,
               this->numberOfSlices,
               this->chunksInSlice,
               this->desiredChunkSize,
               this->segmentToChunkMapping.getView(),
               this->segmentToSliceMapping.getView(),
               this->chunksToSegmentsMapping.getView(),
               this->segmentPointers.getView(),
               this->slices.getView() );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
template< typename SegmentsSizes >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::resolveSliceSizes( SegmentsSizes& segmentsSizes )
{
   // Iterate over segments and allocate slices so that each slice has
   // approximately the same number of allocated elements
   const Index desiredElementsInSlice = this->chunksInSlice * this->desiredChunkSize;

   Index segmentIdx = 0;
   Index sliceSize = 0;
   Index allocatedElementsInSlice = 0;
   this->numberOfSlices = 0;
   while( segmentIdx < segmentsSizes.getSize() ) {
      // Add one segment to the current slice until we reach the desired
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
   /* Now, compute the number of chunks per each segment. Each segment gets one chunk
    * by default. Then each segment will get additional chunks with respect to the
    * number of the elements in the segment. If there are some free chunks left,
    * repeat it again.
    */
   const Index sliceSize = this->slices[ sliceIndex ].size;
   const Index sliceBegin = this->slices[ sliceIndex ].firstSegment;
   const Index allocatedElementsInSlice = this->slices[ sliceIndex ].pointer;
   const Index sliceEnd = sliceBegin + sliceSize;

   Index freeChunks = this->chunksInSlice - sliceSize;
   for( Index i = sliceBegin; i < sliceEnd; i++ )
      this->segmentToChunkMapping.setElement( i, 1 );

   int totalAddedChunks = 0;
   int maxSegmentLength( segmentsSizes[ sliceBegin ] );
   for( Index i = sliceBegin; i < sliceEnd; i++ ) {
      double segmentRatio = 0.0;
      if( allocatedElementsInSlice != 0 )
         segmentRatio = (double) segmentsSizes[ i ] / (double) allocatedElementsInSlice;
      const Index addedChunks = freeChunks * segmentRatio;
      totalAddedChunks += addedChunks;
      this->segmentToChunkMapping[ i ] += addedChunks;
      if( maxSegmentLength < segmentsSizes[ i ] )
         maxSegmentLength = segmentsSizes[ i ];
   }
   TNL_ASSERT_GE( freeChunks, totalAddedChunks, "" );
   freeChunks -= totalAddedChunks;
   while( freeChunks )
      for( Index i = sliceBegin; i < sliceEnd && freeChunks; i++ )
         if( segmentsSizes[ i ] == maxSegmentLength ) {
            this->segmentToChunkMapping[ i ]++;
            freeChunks--;
         }

   // Compute the chunk size
   Index maxChunkInSlice = 0;
   for( Index i = sliceBegin; i < sliceEnd; i++ ) {
      TNL_ASSERT_NE( this->segmentToChunkMapping[ i ], 0, "" );
      maxChunkInSlice = TNL::max( maxChunkInSlice, roundUpDivision( segmentsSizes[ i ], this->segmentToChunkMapping[ i ] ) );
   }

   // Set up the slice info.
   this->slices[ sliceIndex ].chunkSize = maxChunkInSlice;
   this->slices[ sliceIndex ].pointer = elementsToAllocation;
   elementsToAllocation += this->chunksInSlice * maxChunkInSlice;

   for( Index i = sliceBegin; i < sliceEnd; i++ )
      this->segmentToSliceMapping[ i ] = sliceIndex;

   for( Index i = sliceBegin; i < sliceEnd; i++ ) {
      this->segmentPointers[ i + 1 ] = maxChunkInSlice * segmentToChunkMapping[ i ];
      TNL_ASSERT_GE( this->segmentPointers[ i ], 0, "" );
      TNL_ASSERT_GE( this->segmentPointers[ i + 1 ], 0, "" );
   }

   // Finish the segment to chunk mapping by computing the prefix sum.
   for( Index j = sliceBegin + 1; j < sliceEnd; j++ )
      segmentToChunkMapping[ j ] += segmentToChunkMapping[ j - 1 ];
   return true;
}

}  // namespace TNL::Algorithms::Segments
