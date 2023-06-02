// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Allocators/Default.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/SlicedEllpackView.h>
#include <TNL/Algorithms/Segments/SegmentView.h>

namespace TNL::Algorithms::Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int SliceSize = 32 >
class SlicedEllpack
{
public:
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;
   using OffsetsContainer = Containers::Vector< Index, DeviceType, IndexType, IndexAllocator >;
   using ConstOffsetsView = typename OffsetsContainer::ConstViewType;
   [[nodiscard]] static constexpr int
   getSliceSize()
   {
      return SliceSize;
   }
   [[nodiscard]] static constexpr ElementsOrganization
   getOrganization()
   {
      return Organization;
   }
   using ViewType = SlicedEllpackView< Device, Index, Organization, SliceSize >;
   template< typename Device_, typename Index_ >
   using ViewTemplate = SlicedEllpackView< Device_, Index_, Organization, SliceSize >;
   using ConstViewType = SlicedEllpackView< Device, std::add_const_t< Index >, Organization, SliceSize >;
   using SegmentViewType = SegmentView< IndexType, Organization >;

   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return true;
   }

   SlicedEllpack() = default;

   template< typename SizesContainer >
   SlicedEllpack( const SizesContainer& sizes );

   template< typename ListIndex >
   SlicedEllpack( const std::initializer_list< ListIndex >& segmentsSizes );

   SlicedEllpack( const SlicedEllpack& segments ) = default;

   SlicedEllpack( SlicedEllpack&& segments ) noexcept = default;

   [[nodiscard]] static std::string
   getSerializationType();

   [[nodiscard]] static String
   getSegmentsType();

   [[nodiscard]] ViewType
   getView();

   [[nodiscard]] ConstViewType
   getConstView() const;

   /**
    * \brief Set sizes of particular segments.
    */
   template< typename SizesHolder = OffsetsContainer >
   void
   setSegmentsSizes( const SizesHolder& sizes );

   void
   reset();

   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentsCount() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentSize( IndexType segmentIdx ) const;

   /**
    * \brief Number segments.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getStorageSize() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getGlobalIndex( Index segmentIdx, Index localIdx ) const;

   [[nodiscard]] __cuda_callable__
   SegmentViewType
   getSegmentView( IndexType segmentIdx ) const;

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSliceSegmentSizesView() const;

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSliceOffsetsView() const;

   /***
    * \brief Go over all segments and for each segment element call
    * function 'f' with arguments 'args'. The return type of 'f' is bool.
    * When its true, the for-loop continues. Once 'f' returns false, the for-loop
    * is terminated.
    */
   template< typename Function >
   void
   forElements( IndexType first, IndexType last, Function&& f ) const;

   template< typename Function >
   void
   forAllElements( Function&& f ) const;

   template< typename Function >
   void
   forSegments( IndexType begin, IndexType end, Function&& f ) const;

   template< typename Function >
   void
   forAllSegments( Function&& f ) const;

   SlicedEllpack&
   operator=( const SlicedEllpack& source ) = default;

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
   SlicedEllpack&
   operator=( const SlicedEllpack< Device_, Index_, IndexAllocator_, Organization_, SliceSize >& source );

   void
   save( File& file ) const;

   void
   load( File& file );

protected:
   IndexType size = 0;
   IndexType alignedSize = 0;
   IndexType segmentsCount = 0;

   OffsetsContainer sliceOffsets, sliceSegmentSizes;
};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int SliceSize >
std::ostream&
operator<<( std::ostream& str, const SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >& segments )
{
   return printSegments( str, segments );
}

}  // namespace TNL::Algorithms::Segments

#include <TNL/Algorithms/Segments/SlicedEllpack.hpp>
