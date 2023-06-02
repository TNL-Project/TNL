// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/EllpackView.h>
#include <TNL/Algorithms/Segments/SegmentView.h>

namespace TNL::Algorithms::Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int Alignment = 32 >
class Ellpack
{
public:
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;
   [[nodiscard]] static constexpr int
   getAlignment()
   {
      return Alignment;
   }
   [[nodiscard]] static constexpr ElementsOrganization
   getOrganization()
   {
      return Organization;
   }
   using OffsetsContainer = Containers::Vector< IndexType, DeviceType, IndexType >;
   using SegmentsSizes = OffsetsContainer;
   template< typename Device_, typename Index_ >
   using ViewTemplate = EllpackView< Device_, Index_, Organization, Alignment >;
   using ViewType = EllpackView< Device, Index, Organization, Alignment >;
   using ConstViewType = typename ViewType::ConstViewType;
   using SegmentViewType = SegmentView< IndexType, Organization >;

   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return true;
   }

   Ellpack() = default;

   template< typename SizesContainer >
   Ellpack( const SizesContainer& sizes );

   template< typename ListIndex >
   Ellpack( const std::initializer_list< ListIndex >& segmentsSizes );

   Ellpack( IndexType segmentsCount, IndexType segmentSize );

   Ellpack( const Ellpack& segments ) = default;

   Ellpack( Ellpack&& segments ) noexcept = default;

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
   setSegmentsSizes( IndexType segmentsCount, IndexType segmentSize );

   void
   reset();

   /**
    * \brief Number segments.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentsCount() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentSize( IndexType segmentIdx ) const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const;

   /**
    * \brief Returns number of elements that needs to be allocated by a container connected to this segments.
    *
    * \return size of container connected to this segments.
    */
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
   IndexType
   getAlignedSize() const;

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

   Ellpack&
   operator=( const Ellpack& source ) = default;

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_, int Alignment_ >
   Ellpack&
   operator=( const Ellpack< Device_, Index_, IndexAllocator_, Organization_, Alignment_ >& source );

   void
   save( File& file ) const;

   void
   load( File& file );

protected:
   IndexType segmentSize = 0;
   IndexType size = 0;
   IndexType alignedSize = 0;
};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
std::ostream&
operator<<( std::ostream& str, const Ellpack< Device, Index, IndexAllocator, Organization, Alignment >& segments )
{
   return printSegments( str, segments );
}

}  // namespace TNL::Algorithms::Segments

#include <TNL/Algorithms/Segments/Ellpack.hpp>
