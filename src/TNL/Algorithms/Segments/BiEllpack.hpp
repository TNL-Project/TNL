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
: segmentsPermutation( segments.segmentsPermutation ),
  groupPointers( segments.groupPointers )
{
   // update the base
   Base::bind( segments.getElementCount(),
               segments.getStorageSize(),
               this->segmentsPermutation.getView(),
               this->groupPointers.getView() );
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
   this->segmentsPermutation = segments.segmentsPermutation;
   this->groupPointers = segments.groupPointers;
   // update the base
   Base::bind( segments.getElementCount(),
               segments.getStorageSize(),
               this->segmentsPermutation.getView(),
               this->groupPointers.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >&
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::operator=( BiEllpack&& segments ) noexcept( false )
{
   this->segmentsPermutation = std::move( segments.segmentsPermutation );
   this->groupPointers = std::move( segments.groupPointers );
   // update the base
   Base::bind( segments.getElementCount(),
               segments.getStorageSize(),
               this->segmentsPermutation.getView(),
               this->groupPointers.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >&
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::operator=(
   const BiEllpack< Device_, Index_, IndexAllocator_, Organization_, WarpSize >& segments )
{
   this->segmentsPermutation = segments.getSegmentsPermutationView();
   this->groupPointers = segments.getGroupPointersView();
   // update the base
   Base::bind( segments.getElementCount(),
               segments.getStorageSize(),
               this->segmentsPermutation.getView(),
               this->groupPointers.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
typename BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::ViewType
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::getView()
{
   return { this->getElementCount(), this->getStorageSize(), this->getSegmentsPermutationView(), this->getGroupPointersView() };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
auto
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::getConstView() const -> ConstViewType
{
   return { this->getElementCount(), this->getStorageSize(), this->getSegmentsPermutationView(), this->getGroupPointersView() };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::setSegmentsSizes( const SizesHolder& segmentsSizes )
{
   /***
    * BiEllpack implements abstraction of a sparse matrix format from the paper https://doi.org/10.1016/j.jpdc.2014.03.002
    *
    * Here we briefly summarize the main idea of the format. Note, that each segment represents slots for the non-zero matrix
    * elements.
    *
    * 1. We first split all segments into strips of size equal to the warp-size which is 32. If the number of segments is not
    * divisible by the warp-size, we add virtual segments. The number of all segments including the virtual ones can be obtained
    * by calling the getVirtualSegments() method.
    *
    * 2. In the next step we sort segments in each strip in the descending order according to the segments sizes. This changes
    * the ordering of the segments and so we need to store a permutation mapping the original segment index to the new one.
    *  This permutation is stored in the segmentsPermutation array. It means that
    *
    * ```cpp
    * new_segment_idx = segmentsPermutation[ original_segment_idx ]
    *```
    *
    * This array is initiated in the initSegmentsPermutation() method.
    * 3. Next we split each strip of segments into several groups. The number of groups is equal to the log2 of the warp size.
    * For the simplicity, in the following example we assume that the warp-size is 8 and so each strip consists of 8 segments.
    * Assume that the strip is sorted in the descending order according to the segments sizes and it looks as follows
    *
    *  0: * * * * * * * * * * * *
    *  1: * * * * * * * * *
    *  2: * * * * * *
    *  3: * * * *
    *  4: * * *
    *  5: * *
    *  6: * *
    *  7: * *
    *
    * The stars represent the segments. Each group stores several columns of the slots of the segments. Their width is defined
    * as follows:
    * - The width of the first group (with index 0) is equal to the size of the segment number 8/2=4.
    * - The width of the second group (with index 1) is given by the size of the segment number 4/2=2.
    * - The width of the third group (with index 2) is given by the size of the segment number 2/2=1.
    * - The width of the last group (with index 3) is given by the size of the segment number 1/2=0.
    *
    * The following figure shows how the slots of the segments are distributed among the groups.
    *  0: 0 0 0 1 1 1 2 2 2 3 3 3
    *  1: 0 0 0 1 1 1 2 2 2
    *  2: 0 0 0 1 1 1
    *  3: 0 0 0 1 . .
    *  4: 0 0 0
    *  5: 0 0 .
    *  6: 0 0 .
    *  7: 0 0 .
    *
    * The dots represent padding slots in the segments. Note that:
    * - the first group (with index 0) manages slots/elements of all 8 segments
    * - the second group (with index 1) manages slots/elements of the first 4 segments
    * - the third group (with index 2) manages slots/elements of the first 2 segments
    * - the last group (with index 3) manages slots/elements of the first segment
    *
    * In the memory, we first store the elements of the first group (including the padding slots) and then
    * the elements of the second group and so on. The elements are stored either in row-major or column-major
    * order depending on the ElementsOrganization. The offsets of the groups are stored in the groupPointers
    * array which is initiated in the initGroupPointers() method. The number of the slots managed by the group is given
    * by the difference of the offsets of the current group and the subsequent one, i.e.
    *
    * ```cpp
    * groupSize = groupPointers[ groupIdx + 1 ] - groupPointers[ groupIdx ]
    * ```
    *
    * The number of segments managed by the group is given by its local index within the strip
    * (i.e. 0, 1, 2 and 3 in our example). And the width of the group is given by the groupSize divided by the number of
    * segments managed by the group.
    */
   TNL_ASSERT_TRUE( TNL::all( greaterEqual( segmentsSizes, 0 ) ), "Segment size cannot be negative" );
   if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > ) {
      // NOTE: the following functions (e.g. getVirtualSegments and performSegmentBubbleSort)
      // depend on this->size being set
      const Index segmentsCount = segmentsSizes.getSize();
      this->segmentsPermutation.setSize( segmentsCount );
      this->size = sum( segmentsSizes );
      const Index strips = this->getVirtualSegments( segmentsCount ) / Base::getWarpSize();
      this->groupPointers.setSize( strips * ( Base::getLogWarpSize() + 1 ) + 1 );
      this->groupPointers = 0;

      this->initSegmentsPermutation( segmentsSizes );
      this->initGroupPointers( segmentsSizes );
      inplaceExclusiveScan( this->groupPointers );

      const Index storageSize = this->groupPointers.getElement( strips * ( Base::getLogWarpSize() + 1 ) );
      // update the base
      Base::bind( this->size, storageSize, this->segmentsPermutation.getView(), this->groupPointers.getView() );

#ifndef NDEBUG
      // The tests can be called only after updating the VectorViews in the base
      this->verifySegmentPerm( segmentsSizes );
      this->verifySegmentLengths( segmentsSizes );
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
   segmentsPermutation.reset();
   groupPointers.reset();

   // update the base
   Base::bind( 0, 0, this->segmentsPermutation.getView(), this->groupPointers.getView() );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file << this->segmentsPermutation << this->groupPointers;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file >> this->segmentsPermutation >> this->groupPointers;

   // update the base
   Base::bind( this->size, this->storageSize, this->segmentsPermutation.getView(), this->groupPointers.getView() );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::initSegmentsPermutation( const SizesHolder& segmentsSizes )
{
   static_assert( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential >,
                  "The initiation of the segmentPermutationArray can be done only on the CPU." );

   // TODO: The following function could be probably replaced with general sorting algorithms (e.g. bitonnic sort) and run on
   // the GPU.
   const Index segmentsCount = segmentsSizes.getSize();
   if( segmentsCount == 0 )
      return;

   for( Index i = 0; i < segmentsCount; i++ )
      this->segmentsPermutation[ i ] = i;

   const Index strips = this->getVirtualSegments( segmentsCount ) / Base::getWarpSize();
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
               if( this->segmentsPermutation[ k ] == j )
                  permIndex1 = k;
               if( this->segmentsPermutation[ k ] == j + 1 )
                  permIndex2 = k;
            }
            if( segmentsSizes[ permIndex1 ] < segmentsSizes[ permIndex2 ] ) {
               TNL::swap( this->segmentsPermutation[ permIndex1 ], this->segmentsPermutation[ permIndex2 ] );
               sorted = false;
            }
         }
         for( Index j = end - 1 - offset; j > begin + offset; j-- ) {
            for( Index k = begin; k < end + 1; k++ ) {
               if( this->segmentsPermutation[ k ] == j )
                  permIndex1 = k;
               if( this->segmentsPermutation[ k ] == j - 1 )
                  permIndex2 = k;
            }
            if( segmentsSizes[ permIndex2 ] < segmentsSizes[ permIndex1 ] ) {
               TNL::swap( this->segmentsPermutation[ permIndex1 ], this->segmentsPermutation[ permIndex2 ] );
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
   Index numberOfStrips = this->getVirtualSegments( totalSegmentsCount ) / Base::getWarpSize();
   auto groupPointersView = this->groupPointers.getView();
   auto segmentsPermutationView = this->segmentsPermutation.getView();
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
   Algorithms::parallelFor< Device >( 0, this->getVirtualSegments( totalSegmentsCount ) / Base::getWarpSize(), createGroups );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::verifySegmentPerm( const SizesHolder& segmentsSizes )
{
   bool ok = true;
   Index numberOfStrips = this->getVirtualSegments() / Base::getWarpSize();
   for( Index strip = 0; strip < numberOfStrips; strip++ ) {
      Index begin = strip * Base::getWarpSize();
      Index end = ( strip + 1 ) * Base::getWarpSize();
      if( this->getSegmentCount() < end )
         end = this->getSegmentCount();
      for( Index i = begin; i < end - 1; i++ ) {
         Index permIndex1 = 0;
         Index permIndex2 = 0;
         bool first = false;
         bool second = false;
         for( Index j = begin; j < end; j++ ) {
            if( this->segmentsPermutation.getElement( j ) == i ) {
               permIndex1 = j;
               first = true;
            }
            if( this->segmentsPermutation.getElement( j ) == i + 1 ) {
               permIndex2 = j;
               second = true;
            }
         }
         if( ! first || ! second )
            std::cout << "Wrong permutation!\n";
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
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::verifySegmentLengths( const SizesHolder& segmentsSizes )
{
   for( Index segmentIdx = 0; segmentIdx < this->getSegmentCount(); segmentIdx++ ) {
      const Index strip = segmentIdx / Base::getWarpSize();
      const Index stripLength = this->getStripLength( strip );
      const Index groupBegin = ( Base::getLogWarpSize() + 1 ) * strip;
      const Index segmentStripPerm = this->segmentsPermutation.getElement( segmentIdx ) - strip * Base::getWarpSize();
      const Index begin = this->groupPointers.getElement( groupBegin ) * Base::getWarpSize() + segmentStripPerm * stripLength;
      Index elementPtr = begin;
      Index segmentLength = 0;
      const Index groupsCount = detail::BiEllpack< Index, Device, Organization, WarpSize >::getActiveGroupsCount(
         this->segmentsPermutation.getConstView(), segmentIdx );
      for( Index group = 0; group < groupsCount; group++ ) {
         const Index groupSize =
            detail::BiEllpack< Index, Device, Organization, WarpSize >::getGroupSize( this->groupPointers, strip, group );
         for( Index i = 0; i < groupSize; i++ ) {
            Index biElementPtr = elementPtr;
            for( Index j = 0; j < discretePow( (Index) 2, group ); j++ ) {
               segmentLength++;
               biElementPtr += discretePow( (Index) 2, Base::getLogWarpSize() - group ) * stripLength;
            }
            elementPtr++;
         }
      }
      if( segmentsSizes.getElement( segmentIdx ) > segmentLength )
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
