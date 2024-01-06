// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/scan.h>

#include "BiEllpack.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::BiEllpack( const BiEllpack& segments )
: rowPermArray( segments.rowPermArray ), groupPointers( segments.groupPointers )
{
   // update the base
   Base::bind( segments.getSize(), segments.getStorageSize(), this->rowPermArray.getView(), this->groupPointers.getView() );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesContainer >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::BiEllpack( const SizesContainer& segmentsSizes )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename ListIndex >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::BiEllpack(
   const std::initializer_list< ListIndex >& segmentsSizes )
{
   this->setSegmentsSizes( OffsetsContainer( segmentsSizes ) );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >&
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::operator=( const BiEllpack& segments )
{
   this->rowPermArray = segments.rowPermArray;
   this->groupPointers = segments.groupPointers;
   // update the base
   Base::bind( segments.getSize(), segments.getStorageSize(), this->rowPermArray.getView(), this->groupPointers.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >&
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::operator=( BiEllpack&& segments ) noexcept( false )
{
   this->rowPermArray = std::move( segments.rowPermArray );
   this->groupPointers = std::move( segments.groupPointers );
   // update the base
   Base::bind( segments.getSize(), segments.getStorageSize(), this->rowPermArray.getView(), this->groupPointers.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >&
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::operator=(
   const BiEllpack< Device_, Index_, IndexAllocator_, Organization_, WarpSize >& segments )
{
   this->rowPermArray = segments.getRowPermArrayView();
   this->groupPointers = segments.getGroupPointersView();
   // update the base
   Base::bind( segments.getSize(), segments.getStorageSize(), this->rowPermArray.getView(), this->groupPointers.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
typename BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::ViewType
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::getView()
{
   return { this->getSize(), this->getStorageSize(), this->getRowPermArrayView(), this->getGroupPointersView() };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
auto
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::getConstView() const -> ConstViewType
{
   return { this->getSize(), this->getStorageSize(), this->getRowPermArrayView(), this->getGroupPointersView() };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::setSegmentsSizes( const SizesHolder& segmentsSizes )
{
   if constexpr( std::is_same< Device, Devices::Host >::value ) {
      // NOTE: the following functions (e.g. getVirtualRows and performRowBubbleSort)
      // depend on this->size being set
      this->size = segmentsSizes.getSize();
      const Index strips = this->getVirtualRows() / Base::getWarpSize();
      this->rowPermArray.setSize( this->size );
      this->groupPointers.setSize( strips * ( Base::getLogWarpSize() + 1 ) + 1 );
      this->groupPointers = 0;

      this->performRowBubbleSort( segmentsSizes );
      this->computeColumnSizes( segmentsSizes );

      inplaceExclusiveScan( this->groupPointers );

      this->verifyRowPerm( segmentsSizes );
      // TODO: I am not sure what this test is doing.
      // this->verifyRowLengths( segmentsSizes );

      const Index storageSize = Base::getWarpSize() * this->groupPointers.getElement( strips * ( Base::getLogWarpSize() + 1 ) );

      // update the base
      Base::bind( this->size, storageSize, this->rowPermArray.getView(), this->groupPointers.getView() );
   }
   else {
      BiEllpack< Devices::Host, Index, typename Allocators::Default< Devices::Host >::template Allocator< Index >, Organization >
         hostSegments;
      Containers::Vector< Index, Devices::Host, Index > hostSegmentsSizes;
      hostSegmentsSizes = segmentsSizes;
      hostSegments.setSegmentsSizes( hostSegmentsSizes );
      *this = hostSegments;
   }
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::reset()
{
   rowPermArray.reset();
   groupPointers.reset();

   // update the base
   Base::bind( 0, 0, this->rowPermArray.getView(), this->groupPointers.getView() );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file << this->rowPermArray << this->groupPointers;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file >> this->rowPermArray >> this->groupPointers;

   // update the base
   Base::bind( this->size, this->storageSize, this->rowPermArray.getView(), this->groupPointers.getView() );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::performRowBubbleSort( const SizesHolder& segmentsSizes )
{
   if( segmentsSizes.getSize() == 0 )
      return;

   this->rowPermArray.forAllElements(
      [] __cuda_callable__( const Index idx, Index& value )
      {
         value = idx;
      } );

   // if constexpr( std::is_same< DeviceType, Devices::Host >::value )
   {
      const Index strips = this->getVirtualRows() / Base::getWarpSize();
      for( Index i = 0; i < strips; i++ ) {
         Index begin = i * Base::getWarpSize();
         Index end = ( i + 1 ) * Base::getWarpSize() - 1;
         if( this->getSize() - 1 < end )
            end = this->getSize() - 1;
         bool sorted = false;
         Index permIndex1 = 0;
         Index permIndex2 = 0;
         Index offset = 0;
         while( ! sorted ) {
            sorted = true;
            for( Index j = begin + offset; j < end - offset; j++ ) {
               for( Index k = begin; k < end + 1; k++ ) {
                  if( this->rowPermArray.getElement( k ) == j )
                     permIndex1 = k;
                  if( this->rowPermArray.getElement( k ) == j + 1 )
                     permIndex2 = k;
               }
               if( segmentsSizes.getElement( permIndex1 ) < segmentsSizes.getElement( permIndex2 ) ) {
                  Index temp = this->rowPermArray.getElement( permIndex1 );
                  this->rowPermArray.setElement( permIndex1, this->rowPermArray.getElement( permIndex2 ) );
                  this->rowPermArray.setElement( permIndex2, temp );
                  sorted = false;
               }
            }
            for( Index j = end - 1 - offset; j > begin + offset; j-- ) {
               for( Index k = begin; k < end + 1; k++ ) {
                  if( this->rowPermArray.getElement( k ) == j )
                     permIndex1 = k;
                  if( this->rowPermArray.getElement( k ) == j - 1 )
                     permIndex2 = k;
               }
               if( segmentsSizes.getElement( permIndex2 ) < segmentsSizes.getElement( permIndex1 ) ) {
                  Index temp = this->rowPermArray.getElement( permIndex1 );
                  this->rowPermArray.setElement( permIndex1, this->rowPermArray.getElement( permIndex2 ) );
                  this->rowPermArray.setElement( permIndex2, temp );
                  sorted = false;
               }
            }
            offset++;
         }
      }
   }
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::computeColumnSizes( const SizesHolder& segmentsSizes )
{
   Index numberOfStrips = this->getVirtualRows() / Base::getWarpSize();
   auto groupPointersView = this->groupPointers.getView();
   auto segmentsPermutationView = this->rowPermArray.getView();
   auto segmentsSizesView = segmentsSizes.getConstView();
   const Index size = this->getSize();
   auto createGroups = [ = ] __cuda_callable__( const Index strip ) mutable
   {
      Index firstSegment = strip * Base::getWarpSize();
      Index groupBegin = strip * ( Base::getLogWarpSize() + 1 );
      Index emptyGroups = 0;

      // The last strip can be shorter
      if( strip == numberOfStrips - 1 ) {
         Index segmentsCount = size - firstSegment;
         while( segmentsCount <= TNL::pow( 2, Base::getLogWarpSize() - 1 - emptyGroups ) - 1 )
            emptyGroups++;
         for( Index group = groupBegin; group < groupBegin + emptyGroups; group++ )
            groupPointersView[ group ] = 0;
      }

      Index allocatedColumns = 0;
      for( Index groupIdx = emptyGroups; groupIdx < Base::getLogWarpSize(); groupIdx++ ) {
         Index segmentIdx = TNL::pow( 2, Base::getLogWarpSize() - 1 - groupIdx ) - 1;
         Index permSegm = 0;
         while( segmentsPermutationView[ permSegm + firstSegment ] != segmentIdx + firstSegment )
            permSegm++;
         const Index groupWidth = segmentsSizesView[ permSegm + firstSegment ] - allocatedColumns;
         const Index groupHeight = TNL::pow( 2, Base::getLogWarpSize() - groupIdx );
         const Index groupSize = groupWidth * groupHeight;
         allocatedColumns = segmentsSizesView[ permSegm + firstSegment ];
         groupPointersView[ groupIdx + groupBegin ] = groupSize;
      }
   };
   Algorithms::parallelFor< Device >( 0, this->getVirtualRows() / Base::getWarpSize(), createGroups );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::verifyRowPerm( const SizesHolder& segmentsSizes )
{
   bool ok = true;
   Index numberOfStrips = this->getVirtualRows() / Base::getWarpSize();
   for( Index strip = 0; strip < numberOfStrips; strip++ ) {
      Index begin = strip * Base::getWarpSize();
      Index end = ( strip + 1 ) * Base::getWarpSize();
      if( this->getSize() < end )
         end = this->getSize();
      for( Index i = begin; i < end - 1; i++ ) {
         Index permIndex1 = 0;
         Index permIndex2 = 0;
         bool first = false;
         bool second = false;
         for( Index j = begin; j < end; j++ ) {
            if( this->rowPermArray.getElement( j ) == i ) {
               permIndex1 = j;
               first = true;
            }
            if( this->rowPermArray.getElement( j ) == i + 1 ) {
               permIndex2 = j;
               second = true;
            }
         }
         if( ! first || ! second )
            std::cout << "Wrong permutation!" << std::endl;
         if( segmentsSizes.getElement( permIndex1 ) >= segmentsSizes.getElement( permIndex2 ) )
            continue;
         else
            ok = false;
      }
   }
   if( ! ok )
      throw( std::logic_error( "Segments permutation verification failed." ) );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::verifyRowLengths( const SizesHolder& segmentsSizes )
{
   std::cerr << "segmentsSizes = " << segmentsSizes << std::endl;
   for( Index segmentIdx = 0; segmentIdx < this->getSize(); segmentIdx++ ) {
      const Index strip = segmentIdx / Base::getWarpSize();
      const Index stripLength = this->getStripLength( strip );
      const Index groupBegin = ( Base::getLogWarpSize() + 1 ) * strip;
      const Index rowStripPerm = this->rowPermArray.getElement( segmentIdx ) - strip * Base::getWarpSize();
      const Index begin = this->groupPointers.getElement( groupBegin ) * Base::getWarpSize() + rowStripPerm * stripLength;
      Index elementPtr = begin;
      Index rowLength = 0;
      const Index groupsCount = detail::BiEllpack< Index, Device, Organization, WarpSize >::getActiveGroupsCount(
         this->rowPermArray.getConstView(), segmentIdx );
      for( Index group = 0; group < groupsCount; group++ ) {
         const Index groupSize = detail::BiEllpack< Index, Device, Organization, WarpSize >::getGroupSize( strip, group );
         for( Index i = 0; i < groupSize; i++ ) {
            Index biElementPtr = elementPtr;
            for( Index j = 0; j < discretePow( 2, group ); j++ ) {
               rowLength++;
               biElementPtr += discretePow( 2, Base::getLogWarpSize() - group ) * stripLength;
            }
            elementPtr++;
         }
      }
      if( segmentsSizes.getElement( segmentIdx ) > rowLength )
         throw( std::logic_error( "Segments capacities verification failed." ) );
   }
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
auto
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::getStripLength( Index strip ) const -> Index
{
   TNL_ASSERT_GE( strip, 0, "" );

   return this->groupPointers.getElement( ( strip + 1 ) * ( Base::getLogWarpSize() + 1 ) )
        - this->groupPointers.getElement( strip * ( Base::getLogWarpSize() + 1 ) );
}

}  // namespace TNL::Algorithms::Segments
