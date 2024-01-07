// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/DiscreteMath.h>

#include "BiEllpack.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::BiEllpack( const BiEllpack& segments )
: rowsPermutation( segments.rowsPermutation ), groupPointers( segments.groupPointers )
{
   // update the base
   Base::bind( segments.getSize(), segments.getStorageSize(), this->rowsPermutation.getView(), this->groupPointers.getView() );
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
   this->rowsPermutation = segments.rowsPermutation;
   this->groupPointers = segments.groupPointers;
   // update the base
   Base::bind( segments.getSize(), segments.getStorageSize(), this->rowsPermutation.getView(), this->groupPointers.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >&
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::operator=( BiEllpack&& segments ) noexcept( false )
{
   this->rowsPermutation = std::move( segments.rowsPermutation );
   this->groupPointers = std::move( segments.groupPointers );
   // update the base
   Base::bind( segments.getSize(), segments.getStorageSize(), this->rowsPermutation.getView(), this->groupPointers.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >&
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::operator=(
   const BiEllpack< Device_, Index_, IndexAllocator_, Organization_, WarpSize >& segments )
{
   this->rowsPermutation = segments.getrowsPermutationView();
   this->groupPointers = segments.getGroupPointersView();
   // update the base
   Base::bind( segments.getSize(), segments.getStorageSize(), this->rowsPermutation.getView(), this->groupPointers.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
typename BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::ViewType
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::getView()
{
   return { this->getSize(), this->getStorageSize(), this->getrowsPermutationView(), this->getGroupPointersView() };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
auto
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::getConstView() const -> ConstViewType
{
   return { this->getSize(), this->getStorageSize(), this->getrowsPermutationView(), this->getGroupPointersView() };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::setSegmentsSizes( const SizesHolder& segmentsSizes )
{
   TNL_ASSERT_TRUE( TNL::all( greaterEqual( segmentsSizes, 0 ) ), "Segment size cannot be negative" );
   if constexpr( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Sequential >::value ) {
      // NOTE: the following functions (e.g. getVirtualRows and performRowBubbleSort)
      // depend on this->size being set
      const Index segmentsCount = segmentsSizes.getSize();
      this->rowsPermutation.setSize( segmentsCount );
      this->size = sum( segmentsSizes );
      const Index strips = this->getVirtualRows( segmentsCount ) / Base::getWarpSize();
      this->groupPointers.setSize( strips * ( Base::getLogWarpSize() + 1 ) + 1 );
      this->groupPointers = 0;

      this->initRowsPermutation( segmentsSizes );
      this->initGroupPointers( segmentsSizes );
      inplaceExclusiveScan( this->groupPointers );

      const Index storageSize = this->groupPointers.getElement( strips * ( Base::getLogWarpSize() + 1 ) );
      // update the base
      Base::bind( this->size, storageSize, this->rowsPermutation.getView(), this->groupPointers.getView() );

#ifndef NDEBUG
      // The tests can be called only after updating the VectorViews in the base
      this->verifyRowPerm( segmentsSizes );
      this->verifyRowLengths( segmentsSizes );
#endif
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
   rowsPermutation.reset();
   groupPointers.reset();

   // update the base
   Base::bind( 0, 0, this->rowsPermutation.getView(), this->groupPointers.getView() );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file << this->rowsPermutation << this->groupPointers;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file >> this->rowsPermutation >> this->groupPointers;

   // update the base
   Base::bind( this->size, this->storageSize, this->rowsPermutation.getView(), this->groupPointers.getView() );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::initRowsPermutation( const SizesHolder& segmentsSizes )
{
   static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Sequential >::value,
                  "The initiation of the rowPermutationArray can be done only on the CPU." );

   // TODO: The following function could be probably replaced with general sorting algorithms (e.g. bitonnic sort) and run on
   // the GPU.
   const Index segmentsCount = segmentsSizes.getSize();
   if( segmentsCount == 0 )
      return;

   for( Index i = 0; i < segmentsCount; i++ )
      this->rowsPermutation[ i ] = i;

   const Index strips = this->getVirtualRows( segmentsCount ) / Base::getWarpSize();
   for( Index i = 0; i < strips; i++ ) {
      Index begin = i * Base::getWarpSize();
      Index end = ( i + 1 ) * Base::getWarpSize() - 1;
      if( segmentsCount - 1 < end )
         end = segmentsCount - 1;
      bool sorted = false;
      Index permIndex1 = 0;
      Index permIndex2 = 0;
      Index offset = 0;
      while( ! sorted ) {
         sorted = true;
         for( Index j = begin + offset; j < end - offset; j++ ) {
            for( Index k = begin; k < end + 1; k++ ) {
               if( this->rowsPermutation[ k ] == j )
                  permIndex1 = k;
               if( this->rowsPermutation[ k ] == j + 1 )
                  permIndex2 = k;
            }
            if( segmentsSizes[ permIndex1 ] < segmentsSizes[ permIndex2 ] ) {
               Index temp = this->rowsPermutation[ permIndex1 ];
               this->rowsPermutation[ permIndex1 ] = this->rowsPermutation[ permIndex2 ];
               this->rowsPermutation[ permIndex2 ] = temp;
               sorted = false;
            }
         }
         for( Index j = end - 1 - offset; j > begin + offset; j-- ) {
            for( Index k = begin; k < end + 1; k++ ) {
               if( this->rowsPermutation[ k ] == j )
                  permIndex1 = k;
               if( this->rowsPermutation[ k ] == j - 1 )
                  permIndex2 = k;
            }
            if( segmentsSizes[ permIndex2 ] < segmentsSizes[ permIndex1 ] ) {
               Index temp = this->rowsPermutation[ permIndex1 ];
               this->rowsPermutation[ permIndex1 ] = this->rowsPermutation[ permIndex2 ];
               this->rowsPermutation[ permIndex2 ] = temp;
               sorted = false;
            }
         }
         offset++;
      }
   }
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::initGroupPointers( const SizesHolder& segmentsSizes )
{
   const Index totalSegmentsCount = segmentsSizes.getSize();
   Index numberOfStrips = this->getVirtualRows( totalSegmentsCount ) / Base::getWarpSize();
   auto groupPointersView = this->groupPointers.getView();
   auto segmentsPermutationView = this->rowsPermutation.getView();
   auto segmentsSizesView = segmentsSizes.getConstView();
   auto createGroups = [ = ] __cuda_callable__( const Index strip ) mutable
   {
      Index firstSegment = strip * Base::getWarpSize();
      Index groupBegin = strip * ( Base::getLogWarpSize() + 1 );
      Index emptyGroups = 0;

      // The last strip can be shorter
      if( strip == numberOfStrips - 1 ) {
         Index segmentsCount = totalSegmentsCount - firstSegment;
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
   Algorithms::parallelFor< Device >( 0, this->getVirtualRows( totalSegmentsCount ) / Base::getWarpSize(), createGroups );
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
      if( this->getSegmentsCount() < end )
         end = this->getSegmentsCount();
      for( Index i = begin; i < end - 1; i++ ) {
         Index permIndex1 = 0;
         Index permIndex2 = 0;
         bool first = false;
         bool second = false;
         for( Index j = begin; j < end; j++ ) {
            if( this->rowsPermutation.getElement( j ) == i ) {
               permIndex1 = j;
               first = true;
            }
            if( this->rowsPermutation.getElement( j ) == i + 1 ) {
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
   for( Index segmentIdx = 0; segmentIdx < this->getSegmentsCount(); segmentIdx++ ) {
      const Index strip = segmentIdx / Base::getWarpSize();
      const Index stripLength = this->getStripLength( strip );
      const Index groupBegin = ( Base::getLogWarpSize() + 1 ) * strip;
      const Index rowStripPerm = this->rowsPermutation.getElement( segmentIdx ) - strip * Base::getWarpSize();
      const Index begin = this->groupPointers.getElement( groupBegin ) * Base::getWarpSize() + rowStripPerm * stripLength;
      Index elementPtr = begin;
      Index rowLength = 0;
      const Index groupsCount = detail::BiEllpack< Index, Device, Organization, WarpSize >::getActiveGroupsCount(
         this->rowsPermutation.getConstView(), segmentIdx );
      for( Index group = 0; group < groupsCount; group++ ) {
         const Index groupSize =
            detail::BiEllpack< Index, Device, Organization, WarpSize >::getGroupSize( this->groupPointers, strip, group );
         for( Index i = 0; i < groupSize; i++ ) {
            Index biElementPtr = elementPtr;
            for( Index j = 0; j < discretePow( (Index) 2, group ); j++ ) {
               rowLength++;
               biElementPtr += discretePow( (Index) 2, Base::getLogWarpSize() - group ) * stripLength;
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
   TNL_ASSERT_LE( ( strip + 1 ) * ( Base::getLogWarpSize() + 1 ), this->groupPointers.getSize(), "" );

   return this->groupPointers.getElement( ( strip + 1 ) * ( Base::getLogWarpSize() + 1 ) )
        - this->groupPointers.getElement( strip * ( Base::getLogWarpSize() + 1 ) );
}

}  // namespace TNL::Algorithms::Segments
