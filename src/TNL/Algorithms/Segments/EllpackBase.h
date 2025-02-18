// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "SegmentView.h"
#include "ElementsOrganization.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
class EllpackBase
{
public:
   using DeviceType = Device;

   using IndexType = std::remove_const_t< Index >;

   using SegmentViewType = SegmentView< IndexType, Organization >;

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

   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return true;
   }

   __cuda_callable__
   EllpackBase() = default;

   __cuda_callable__
   EllpackBase( IndexType segmentsCount, IndexType segmentSize, IndexType alignedSize );

   __cuda_callable__
   EllpackBase( const EllpackBase& ) = default;

   __cuda_callable__
   EllpackBase( EllpackBase&& ) noexcept = default;

   EllpackBase&
   operator=( const EllpackBase& ) = delete;

   EllpackBase&
   operator=( EllpackBase&& ) = delete;

   [[nodiscard]] static std::string
   getSerializationType();

   [[nodiscard]] static std::string
   getSegmentsType();

   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentsCount() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentSize( IndexType segmentIdx ) const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentSize() const;

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

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead" )]]
   void
   forElements( IndexType begin, IndexType end, Function&& function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllElements instead" )]]
   void
   forAllElements( Function&& function ) const;

   template< typename Array, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead" )]]
   void
   forElements( const Array& segmentIndexes, Index begin, Index end, Function function ) const;

   template< typename Array, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead" )]]
   void
   forElements( const Array& segmentIndexes, Function function ) const;

   template< typename Condition, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElementsIf instead" )]]
   void
   forElementsIf( IndexType begin, IndexType end, Condition condition, Function function ) const;

   template< typename Condition, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllElementsIf instead" )]]
   void
   forAllElementsIf( Condition condition, Function function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forSegments instead" )]]
   void
   forSegments( IndexType begin, IndexType end, Function&& function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllSegments instead" )]]
   void
   forAllSegments( Function&& function ) const;

   // TODO: sequentialForSegments, sequentialForAllSegments

protected:
   IndexType segmentSize = 0;
   IndexType segmentsCount = 0;
   IndexType alignedSize = 0;

   /**
    * \brief Re-initializes the internal attributes of the base class.
    *
    * Note that this function is \e protected to ensure that the user cannot
    * modify the base class of segments. For the same reason, in future code
    * development we also need to make sure that all non-const functions in
    * the base class return by value and not by reference.
    */
   __cuda_callable__
   void
   bind( IndexType segmentsCount, IndexType segmentSize, IndexType alignedSize );
};

}  // namespace TNL::Algorithms::Segments

#include "EllpackBase.hpp"
#include "print.h"
