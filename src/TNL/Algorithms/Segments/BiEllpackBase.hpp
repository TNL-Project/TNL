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
                                                              OffsetsView segmentsPermutation,
                                                              OffsetsView groupPointers )
{
   this->size = size;
   this->storageSize = storageSize;
   this->segmentsPermutation.bind( std::move( segmentsPermutation ) );
   this->groupPointers.bind( std::move( groupPointers ) );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
BiEllpackBase< Device, Index, Organization, WarpSize >::BiEllpackBase( IndexType size,
                                                                       IndexType storageSize,
                                                                       OffsetsView segmentsPermutation,
                                                                       OffsetsView groupPointers )
: size( size ),
  storageSize( storageSize ),
  segmentsPermutation( std::move( segmentsPermutation ) ),
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
BiEllpackBase< Device, Index, Organization, WarpSize >::getSegmentCount() const -> IndexType
{
   return this->segmentsPermutation.getSize();
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getSegmentsCount() const -> IndexType
{
   return this->getSegmentCount();
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getSegmentSize( IndexType segmentIdx ) const -> IndexType
{
   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentSizeDirect(
         segmentsPermutation, groupPointers, segmentIdx );
#else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentSize(
         segmentsPermutation, groupPointers, segmentIdx );
#endif
   }
   else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentSizeDirect(
         segmentsPermutation, groupPointers, segmentIdx );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getSize() const -> IndexType
{
   return this->getElementCount();
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getElementCount() const -> IndexType
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
   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getGlobalIndexDirect(
         segmentsPermutation, groupPointers, segmentIdx, localIdx );
#else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getGlobalIndex(
         segmentsPermutation, groupPointers, segmentIdx, localIdx );
#endif
   }
   else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getGlobalIndexDirect(
         segmentsPermutation, groupPointers, segmentIdx, localIdx );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getSegmentView( IndexType segmentIdx ) const -> SegmentViewType
{
   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentViewDirect(
         segmentsPermutation, groupPointers, segmentIdx );
#else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentView(
         segmentsPermutation, groupPointers, segmentIdx );
#endif
   }
   else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentViewDirect(
         segmentsPermutation, groupPointers, segmentIdx );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getSegmentsPermutationView() -> OffsetsView
{
   return segmentsPermutation.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getSegmentsPermutationView() const -> ConstOffsetsView
{
   return segmentsPermutation.getConstView();
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
BiEllpackBase< Device, Index, Organization, WarpSize >::getVirtualSegments() const -> IndexType
{
   return this->getVirtualSegments( this->getSegmentCount() );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackBase< Device, Index, Organization, WarpSize >::getVirtualSegments( IndexType segmentsCount ) const -> IndexType
{
   if( segmentsCount % getWarpSize() != 0 )
      return segmentsCount + getWarpSize() - ( segmentsCount % getWarpSize() );
   return segmentsCount;
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
template< typename Function >
void
BiEllpackBase< Device, Index, Organization, WarpSize >::forElements( IndexType begin, IndexType end, Function&& function ) const
{
   const auto segmentsPermutationView = this->segmentsPermutation.getConstView();
   const auto groupPointersView = this->groupPointers.getConstView();
   auto work = [ segmentsPermutationView, groupPointersView, function ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      const IndexType strip = segmentIdx / getWarpSize();
      const IndexType firstGroupInStrip = strip * ( getLogWarpSize() + 1 );
      const IndexType segmentStripPerm = segmentsPermutationView[ segmentIdx ] - strip * getWarpSize();
      const IndexType groupsCount =
         detail::BiEllpack< IndexType, DeviceType, Organization, getWarpSize() >::getActiveGroupsCountDirect(
            segmentsPermutationView, segmentIdx );
      IndexType groupHeight = getWarpSize();
      IndexType localIdx = 0;
      for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
         IndexType groupOffset = groupPointersView[ groupIdx ];
         const IndexType groupSize = groupPointersView[ groupIdx + 1 ] - groupOffset;
         if( groupSize ) {
            const IndexType groupWidth = groupSize / groupHeight;
            for( IndexType i = 0; i < groupWidth; i++ ) {
               if constexpr( Organization == RowMajorOrder ) {
                  function( segmentIdx, localIdx, groupOffset + segmentStripPerm * groupWidth + i );
               }
               else {
                  function( segmentIdx, localIdx, groupOffset + segmentStripPerm + i * groupHeight );
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
   this->forElements( 0, this->getSegmentCount(), function );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
template< typename Array, typename Function >
void
BiEllpackBase< Device, Index, Organization, WarpSize >::forElements( const Array& segmentIndexes,
                                                                     Index begin,
                                                                     Index end,
                                                                     Function function ) const
{
   const auto segmentsPermutationView = this->segmentsPermutation.getConstView();
   const auto groupPointersView = this->groupPointers.getConstView();
   auto segmentIndexesView = segmentIndexes.getConstView();
   auto work =
      [ segmentIndexesView, segmentsPermutationView, groupPointersView, function ] __cuda_callable__( IndexType idx ) mutable
   {
      const IndexType segmentIdx = segmentIndexesView[ idx ];
      const IndexType strip = segmentIdx / getWarpSize();
      const IndexType firstGroupInStrip = strip * ( getLogWarpSize() + 1 );
      const IndexType segmentStripPerm = segmentsPermutationView[ segmentIdx ] - strip * getWarpSize();
      const IndexType groupsCount =
         detail::BiEllpack< IndexType, DeviceType, Organization, getWarpSize() >::getActiveGroupsCountDirect(
            segmentsPermutationView, segmentIdx );
      IndexType groupHeight = getWarpSize();
      IndexType localIdx = 0;
      for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
         IndexType groupOffset = groupPointersView[ groupIdx ];
         const IndexType groupSize = groupPointersView[ groupIdx + 1 ] - groupOffset;
         if( groupSize ) {
            const IndexType groupWidth = groupSize / groupHeight;
            for( IndexType i = 0; i < groupWidth; i++ ) {
               if constexpr( Organization == RowMajorOrder ) {
                  function( segmentIdx, localIdx, groupOffset + segmentStripPerm * groupWidth + i );
               }
               else {
                  function( segmentIdx, localIdx, groupOffset + segmentStripPerm + i * groupHeight );
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
template< typename Array, typename Function >
void
BiEllpackBase< Device, Index, Organization, WarpSize >::forElements( const Array& segmentIndexes, Function function ) const
{
   this->forElements( segmentIndexes, 0, segmentIndexes.getSize(), function );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
template< typename Condition, typename Function >
void
BiEllpackBase< Device, Index, Organization, WarpSize >::forElementsIf( IndexType begin,
                                                                       IndexType end,
                                                                       Condition condition,
                                                                       Function function ) const
{
   const auto segmentsPermutationView = this->segmentsPermutation.getConstView();
   const auto groupPointersView = this->groupPointers.getConstView();
   auto work =
      [ segmentsPermutationView, groupPointersView, condition, function ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      if( ! condition( segmentIdx ) )
         return;
      const IndexType strip = segmentIdx / getWarpSize();
      const IndexType firstGroupInStrip = strip * ( getLogWarpSize() + 1 );
      const IndexType segmentStripPerm = segmentsPermutationView[ segmentIdx ] - strip * getWarpSize();
      const IndexType groupsCount =
         detail::BiEllpack< IndexType, DeviceType, Organization, getWarpSize() >::getActiveGroupsCountDirect(
            segmentsPermutationView, segmentIdx );
      IndexType groupHeight = getWarpSize();
      IndexType localIdx = 0;
      for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
         IndexType groupOffset = groupPointersView[ groupIdx ];
         const IndexType groupSize = groupPointersView[ groupIdx + 1 ] - groupOffset;
         if( groupSize ) {
            const IndexType groupWidth = groupSize / groupHeight;
            for( IndexType i = 0; i < groupWidth; i++ ) {
               if constexpr( Organization == RowMajorOrder ) {
                  function( segmentIdx, localIdx, groupOffset + segmentStripPerm * groupWidth + i );
               }
               else {
                  function( segmentIdx, localIdx, groupOffset + segmentStripPerm + i * groupHeight );
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
template< typename Condition, typename Function >
void
BiEllpackBase< Device, Index, Organization, WarpSize >::forAllElementsIf( Condition condition, Function function ) const
{
   this->forElementsIf( 0, this->getSegmentCount(), condition, function );
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
   this->forSegments( 0, this->getSegmentCount(), function );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
void
BiEllpackBase< Device, Index, Organization, WarpSize >::printStructure( std::ostream& str ) const
{
   const IndexType stripsCount = roundUpDivision( this->getElementCount(), getWarpSize() );
   for( IndexType stripIdx = 0; stripIdx < stripsCount; stripIdx++ ) {
      str << "Strip: " << stripIdx << '\n';
      const IndexType firstGroupIdx = stripIdx * ( getLogWarpSize() + 1 );
      const IndexType lastGroupIdx = firstGroupIdx + getLogWarpSize() + 1;
      IndexType groupHeight = getWarpSize();
      for( IndexType groupIdx = firstGroupIdx; groupIdx < lastGroupIdx; groupIdx++ ) {
         const IndexType groupSize = groupPointers.getElement( groupIdx + 1 ) - groupPointers.getElement( groupIdx );
         const IndexType groupWidth = groupSize / groupHeight;
         str << "\tGroup: " << groupIdx << " size = " << groupSize << " width = " << groupWidth << " height = " << groupHeight
             << " offset = " << groupPointers.getElement( groupIdx ) << '\n';
         groupHeight /= 2;
      }
   }
}

}  // namespace TNL::Algorithms::Segments
