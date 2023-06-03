// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "SegmentView.h"
#include "ElementsOrganization.h"
#include "printSegments.h"

namespace TNL::Algorithms::Segments {

template< typename Device,
          typename Index,
          ElementsOrganization Organization = Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int Alignment = 32 >
class EllpackView
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
   using ViewType = EllpackView;
   using ConstViewType = ViewType;
   using SegmentViewType = SegmentView< IndexType, Organization >;

   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return true;
   }

   __cuda_callable__
   EllpackView() = default;

   __cuda_callable__
   EllpackView( IndexType segmentsCount, IndexType segmentSize, IndexType alignedSize );

   __cuda_callable__
   EllpackView( IndexType segmentsCount, IndexType segmentSize );

   __cuda_callable__
   EllpackView( const EllpackView& ) = default;

   __cuda_callable__
   EllpackView( EllpackView&& ) noexcept = default;

   EllpackView&
   operator=( const EllpackView& ) = delete;

   EllpackView&
   operator=( EllpackView&& ) = delete;

   __cuda_callable__
   void
   bind( EllpackView& view );

   __cuda_callable__
   void
   bind( EllpackView&& view );

   [[nodiscard]] static std::string
   getSerializationType();

   [[nodiscard]] static String
   getSegmentsType();

   [[nodiscard]] __cuda_callable__
   ViewType
   getView();

   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const;

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
   forElements( IndexType begin, IndexType end, Function&& f ) const;

   template< typename Function >
   void
   forAllElements( Function&& f ) const;

   template< typename Function >
   void
   forSegments( IndexType begin, IndexType end, Function&& f ) const;

   template< typename Function >
   void
   forAllSegments( Function&& f ) const;

   void
   save( File& file ) const;

   void
   load( File& file );

protected:
   IndexType segmentSize = 0;
   IndexType segmentsCount = 0;
   IndexType alignedSize = 0;
};

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
std::ostream&
operator<<( std::ostream& str, const EllpackView< Device, Index, Organization, Alignment >& ellpack )
{
   return printSegments( str, ellpack );
}

}  // namespace TNL::Algorithms::Segments

#include "EllpackView.hpp"
