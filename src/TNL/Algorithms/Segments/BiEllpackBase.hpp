// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "BiEllpackBase.h"
#include "detail/BiEllpack.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
void
BiEllpackBase< Device, Index, Organization, WarpSize >::bind( IndexType size,
                                                              IndexType storageSize,
                                                              OffsetsView rowPermArray,
                                                              OffsetsView groupPointers )
{
   this->size = size;
   this->storageSize = storageSize;
   this->rowPermArray.bind( std::move( rowPermArray ) );
   this->groupPointers.bind( std::move( groupPointers ) );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
BiEllpackBase< Device, Index, Organization, WarpSize >::BiEllpackBase( IndexType size,
                                                                       IndexType storageSize,
                                                                       OffsetsView rowPermArray,
                                                                       OffsetsView groupPointers )
: size( size ), storageSize( storageSize ), rowPermArray( std::move( rowPermArray ) ),
  groupPointers( std::move( groupPointers ) )
{}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
std::string
BiEllpackBase< Device, Index, Organization, WarpSize >::getSerializationType()
{
   return "BiEllpack< " + TNL::getSerializationType< IndexType >() + ", " + TNL::getSerializationType( Organization ) + ", "
        + std::to_string( WarpSize ) + " >";
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
std::string
BiEllpackBase< Device, Index, Organization, WarpSize >::getSegmentsType()
{
   return "BiEllpack";
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getSegmentsCount() const -> IndexType
{
   return this->size;
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getSegmentSize( IndexType segmentIdx ) const -> IndexType
{
   if constexpr( std::is_same< DeviceType, Devices::Cuda >::value ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
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
BiEllpackBase< Device, Index, Organization, WarpSize >::getSize() const -> IndexType
{
   return this->size;
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getGlobalIndex( Index segmentIdx, Index localIdx ) const -> IndexType
{
   if constexpr( std::is_same< DeviceType, Devices::Cuda >::value ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
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
BiEllpackBase< Device, Index, Organization, WarpSize >::getSegmentView( IndexType segmentIdx ) const -> SegmentViewType
{
   if constexpr( std::is_same< DeviceType, Devices::Cuda >::value ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
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
BiEllpackBase< Device, Index, Organization, WarpSize >::getRowPermArrayView() -> OffsetsView
{
   return rowPermArray.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getRowPermArrayView() const -> ConstOffsetsView
{
   return rowPermArray.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getGroupPointersView() -> OffsetsView
{
   return groupPointers.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getGroupPointersView() const -> ConstOffsetsView
{
   return groupPointers.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getVirtualRows() const -> IndexType
{
   if( this->size % getWarpSize() != 0 )
      return this->size + getWarpSize() - ( this->size % getWarpSize() );
   return this->size;
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
template< typename Function >
void
BiEllpackBase< Device, Index, Organization, WarpSize >::forElements( IndexType begin, IndexType end, Function&& function ) const
{
   const auto segmentsPermutationView = this->rowPermArray.getConstView();
   const auto groupPointersView = this->groupPointers.getConstView();
   auto work = [ segmentsPermutationView, groupPointersView, function ] __cuda_callable__( IndexType segmentIdx ) mutable
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
      IndexType localIdx = 0;
      for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
         IndexType groupOffset = groupPointersView[ groupIdx ];
         const IndexType groupSize = groupPointersView[ groupIdx + 1 ] - groupOffset;
         // printf( "groupSize = %d \n", groupSize );
         if( groupSize ) {
            const IndexType groupWidth = groupSize / groupHeight;
            for( IndexType i = 0; i < groupWidth; i++ ) {
               if constexpr( Organization == RowMajorOrder ) {
                  function( segmentIdx, localIdx, groupOffset + rowStripPerm * groupWidth + i );
               }
               else {
                  /*printf( "segmentIdx = %d localIdx = %d globalIdx = %d groupIdx = %d groupSize = %d groupWidth = %d\n",
                     segmentIdx, localIdx, groupOffset + rowStripPerm + i * groupHeight,
                     groupIdx, groupSize, groupWidth );*/
                  function( segmentIdx, localIdx, groupOffset + rowStripPerm + i * groupHeight );
               }
               localIdx++;
            }
         }
         groupHeight /= 2;
      }
   };
   Algorithms::parallelFor< DeviceType >( begin, end, work );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
template< typename Function >
void
BiEllpackBase< Device, Index, Organization, WarpSize >::forAllElements( Function&& function ) const
{
   this->forElements( 0, this->getSegmentsCount(), function );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
template< typename Function >
void
BiEllpackBase< Device, Index, Organization, WarpSize >::forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   const auto& self = *this;
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = self.getSegmentView( segmentIdx );
      function( segment );
   };
   TNL::Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
template< typename Function >
void
BiEllpackBase< Device, Index, Organization, WarpSize >::forAllSegments( Function&& function ) const
{
   this->forSegments( 0, this->getSegmentsCount(), function );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
void
BiEllpackBase< Device, Index, Organization, WarpSize >::printStructure( std::ostream& str ) const
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
