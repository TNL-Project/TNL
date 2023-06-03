// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "BiEllpackView.h"
#include "detail/BiEllpack.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
BiEllpackView< Device, Index, Organization, WarpSize >::BiEllpackView( const IndexType size,
                                                                       const IndexType storageSize,
                                                                       const IndexType virtualRows,
                                                                       const OffsetsView& rowPermArray,
                                                                       const OffsetsView& groupPointers )
: size( size ), storageSize( storageSize ), virtualRows( virtualRows ), rowPermArray( rowPermArray ),
  groupPointers( groupPointers )
{}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
BiEllpackView< Device, Index, Organization, WarpSize >::BiEllpackView( const IndexType size,
                                                                       const IndexType storageSize,
                                                                       const IndexType virtualRows,
                                                                       const OffsetsView&& rowPermArray,
                                                                       const OffsetsView&& groupPointers )
: size( size ), storageSize( storageSize ), virtualRows( virtualRows ), rowPermArray( std::move( rowPermArray ) ),
  groupPointers( std::move( groupPointers ) )
{}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
void
BiEllpackView< Device, Index, Organization, WarpSize >::bind( BiEllpackView& view )
{
   this->size = view.size;
   this->storageSize = view.storageSize;
   this->virtualRows = view.virtualRows;
   this->rowPermArray.bind( view.rowPermArray );
   this->groupPointers.bind( view.groupPointers );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
void
BiEllpackView< Device, Index, Organization, WarpSize >::bind( BiEllpackView&& view )
{
   this->size = view.size;
   this->storageSize = view.storageSize;
   this->virtualRows = view.virtualRows;
   this->rowPermArray.bind( view.rowPermArray );
   this->groupPointers.bind( view.groupPointers );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
std::string
BiEllpackView< Device, Index, Organization, WarpSize >::getSerializationType()
{
   return "BiEllpack< " + TNL::getSerializationType< IndexType >() + ", " + TNL::getSerializationType( Organization ) + ", "
        + std::to_string( WarpSize ) + " >";
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
String
BiEllpackView< Device, Index, Organization, WarpSize >::getSegmentsType()
{
   return "BiEllpack";
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
typename BiEllpackView< Device, Index, Organization, WarpSize >::ViewType
BiEllpackView< Device, Index, Organization, WarpSize >::getView()
{
   return { size, storageSize, virtualRows, rowPermArray.getView(), groupPointers.getView() };
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, Organization, WarpSize >::getConstView() const -> ConstViewType
{
   return { size, storageSize, virtualRows, rowPermArray.getConstView(), groupPointers.getConstView() };
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, Organization, WarpSize >::getSegmentsCount() const -> IndexType
{
   return this->size;
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, Organization, WarpSize >::getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
#ifdef __CUDA_ARCH__
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentSizeDirect(
         rowPermArray, groupPointers, segmentIdx );
#else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentSize(
         rowPermArray, groupPointers, segmentIdx );
#endif
   }
   else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentSizeDirect(
         rowPermArray, groupPointers, segmentIdx );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, Organization, WarpSize >::getSize() const -> IndexType
{
   return this->size;
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, Organization, WarpSize >::getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, Organization, WarpSize >::getGlobalIndex( const Index segmentIdx, const Index localIdx ) const
   -> IndexType
{
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
#ifdef __CUDA_ARCH__
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getGlobalIndexDirect(
         rowPermArray, groupPointers, segmentIdx, localIdx );
#else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getGlobalIndex(
         rowPermArray, groupPointers, segmentIdx, localIdx );
#endif
   }
   else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getGlobalIndexDirect(
         rowPermArray, groupPointers, segmentIdx, localIdx );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, Organization, WarpSize >::getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
#ifdef __CUDA_ARCH__
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentViewDirect(
         rowPermArray, groupPointers, segmentIdx );
#else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentView(
         rowPermArray, groupPointers, segmentIdx );
#endif
   }
   else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentViewDirect(
         rowPermArray, groupPointers, segmentIdx );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, Organization, WarpSize >::getRowPermArrayView() const -> ConstOffsetsView
{
   return rowPermArray.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, Organization, WarpSize >::getGroupPointersView() const -> ConstOffsetsView
{
   return groupPointers.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
template< typename Function >
void
BiEllpackView< Device, Index, Organization, WarpSize >::forElements( IndexType first, IndexType last, Function&& f ) const
{
   const auto segmentsPermutationView = this->rowPermArray.getConstView();
   const auto groupPointersView = this->groupPointers.getConstView();
   auto work = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      const IndexType strip = segmentIdx / getWarpSize();
      const IndexType firstGroupInStrip = strip * ( getLogWarpSize() + 1 );
      const IndexType rowStripPerm = segmentsPermutationView[ segmentIdx ] - strip * getWarpSize();
      const IndexType groupsCount =
         detail::BiEllpack< IndexType, DeviceType, Organization, getWarpSize() >::getActiveGroupsCountDirect(
            segmentsPermutationView, segmentIdx );
      IndexType groupHeight = getWarpSize();
      // printf( "segmentIdx = %d strip = %d firstGroupInStrip = %d rowStripPerm = %d groupsCount = %d \n", segmentIdx, strip,
      // firstGroupInStrip, rowStripPerm, groupsCount );
      IndexType localIdx( 0 );
      for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
         IndexType groupOffset = groupPointersView[ groupIdx ];
         const IndexType groupSize = groupPointersView[ groupIdx + 1 ] - groupOffset;
         // printf( "groupSize = %d \n", groupSize );
         if( groupSize ) {
            const IndexType groupWidth = groupSize / groupHeight;
            for( IndexType i = 0; i < groupWidth; i++ ) {
               if( Organization == RowMajorOrder ) {
                  f( segmentIdx, localIdx, groupOffset + rowStripPerm * groupWidth + i );
               }
               else {
                  /*printf( "segmentIdx = %d localIdx = %d globalIdx = %d groupIdx = %d groupSize = %d groupWidth = %d\n",
                     segmentIdx, localIdx, groupOffset + rowStripPerm + i * groupHeight,
                     groupIdx, groupSize, groupWidth );*/
                  f( segmentIdx, localIdx, groupOffset + rowStripPerm + i * groupHeight );
               }
               localIdx++;
            }
         }
         groupHeight /= 2;
      }
   };
   Algorithms::parallelFor< DeviceType >( first, last, work );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
template< typename Function >
void
BiEllpackView< Device, Index, Organization, WarpSize >::forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
template< typename Function >
void
BiEllpackView< Device, Index, Organization, WarpSize >::forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   auto view = this->getConstView();
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = view.getSegmentView( segmentIdx );
      function( segment );
   };
   TNL::Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
template< typename Function >
void
BiEllpackView< Device, Index, Organization, WarpSize >::forAllSegments( Function&& f ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
void
BiEllpackView< Device, Index, Organization, WarpSize >::save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->virtualRows );
   file << this->rowPermArray << this->groupPointers;
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
void
BiEllpackView< Device, Index, Organization, WarpSize >::printStructure( std::ostream& str ) const
{
   const IndexType stripsCount = roundUpDivision( this->getSize(), getWarpSize() );
   for( IndexType stripIdx = 0; stripIdx < stripsCount; stripIdx++ ) {
      str << "Strip: " << stripIdx << std::endl;
      const IndexType firstGroupIdx = stripIdx * ( getLogWarpSize() + 1 );
      const IndexType lastGroupIdx = firstGroupIdx + getLogWarpSize() + 1;
      IndexType groupHeight = getWarpSize();
      for( IndexType groupIdx = firstGroupIdx; groupIdx < lastGroupIdx; groupIdx++ ) {
         const IndexType groupSize = groupPointers.getElement( groupIdx + 1 ) - groupPointers.getElement( groupIdx );
         const IndexType groupWidth = groupSize / groupHeight;
         str << "\tGroup: " << groupIdx << " size = " << groupSize << " width = " << groupWidth << " height = " << groupHeight
             << " offset = " << groupPointers.getElement( groupIdx ) << std::endl;
         groupHeight /= 2;
      }
   }
}

}  // namespace TNL::Algorithms::Segments
