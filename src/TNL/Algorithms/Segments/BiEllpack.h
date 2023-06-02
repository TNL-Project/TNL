// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Allocators/Default.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/BiEllpackView.h>
#include <TNL/Algorithms/Segments/SegmentView.h>

namespace TNL::Algorithms::Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int WarpSize = 32 >
class BiEllpack
{
public:
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;
   using OffsetsContainer = Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocator >;
   using ConstOffsetsView = typename OffsetsContainer::ConstViewType;
   using ViewType = BiEllpackView< Device, Index, Organization, WarpSize >;
   template< typename Device_, typename Index_ >
   using ViewTemplate = BiEllpackView< Device_, Index_, Organization, WarpSize >;
   using ConstViewType = typename ViewType::ConstViewType;
   using SegmentViewType = typename ViewType::SegmentViewType;

   [[nodiscard]] static constexpr ElementsOrganization
   getOrganization()
   {
      return Organization;
   }

   [[nodiscard]] static constexpr int
   getWarpSize()
   {
      return WarpSize;
   }

   [[nodiscard]] static constexpr int
   getLogWarpSize()
   {
      return TNL::discreteLog2( WarpSize );
   }

   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return true;
   }

   BiEllpack() = default;

   template< typename SizesContainer >
   BiEllpack( const SizesContainer& sizes );

   template< typename ListIndex >
   BiEllpack( const std::initializer_list< ListIndex >& segmentsSizes );

   BiEllpack( const BiEllpack& segments ) = default;

   BiEllpack( BiEllpack&& segments ) noexcept = default;

   [[nodiscard]] static std::string
   getSerializationType();

   [[nodiscard]] static String
   getSegmentsType();

   [[nodiscard]] ViewType
   getView();

   [[nodiscard]] ConstViewType
   getConstView() const;

   /**
    * \brief Number of segments.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentsCount() const;

   /**
    * \brief Set sizes of particular segments.
    */
   template< typename SizesHolder = OffsetsContainer >
   void
   setSegmentsSizes( const SizesHolder& sizes );

   void
   reset();

   [[nodiscard]] IndexType
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
   getGlobalIndex( IndexType segmentIdx, IndexType localIdx ) const;

   [[nodiscard]] __cuda_callable__
   SegmentViewType
   getSegmentView( IndexType segmentIdx ) const;

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getRowPermArrayView() const;

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getGroupPointersView() const;

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

   BiEllpack&
   operator=( const BiEllpack& source ) = default;

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
   BiEllpack&
   operator=( const BiEllpack< Device_, Index_, IndexAllocator_, Organization_, WarpSize >& source );

   void
   save( File& file ) const;

   void
   load( File& file );

   void
   printStructure( std::ostream& str ) const;

   // TODO: nvcc needs this public because of lambda function used inside
   template< typename SizesHolder = OffsetsContainer >
   void
   performRowBubbleSort( const SizesHolder& segmentsSize );

   // TODO: the same as  above
   template< typename SizesHolder = OffsetsContainer >
   void
   computeColumnSizes( const SizesHolder& segmentsSizes );

protected:
   template< typename SizesHolder = OffsetsContainer >
   void
   verifyRowPerm( const SizesHolder& segmentsSizes );

   template< typename SizesHolder = OffsetsContainer >
   void
   verifyRowLengths( const SizesHolder& segmentsSizes );

   [[nodiscard]] IndexType
   getStripLength( IndexType stripIdx ) const;

   [[nodiscard]] IndexType
   getGroupLength( IndexType strip, IndexType group ) const;

   IndexType size = 0, storageSize = 0;

   IndexType virtualRows = 0;

   OffsetsContainer rowPermArray;

   OffsetsContainer groupPointers;

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_, int WarpSize_ >
   friend class BiEllpack;
};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
std::ostream&
operator<<( std::ostream& str, const BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >& segments )
{
   return printSegments( str, segments );
}

}  // namespace TNL::Algorithms::Segments

#include <TNL/Algorithms/Segments/BiEllpack.hpp>
