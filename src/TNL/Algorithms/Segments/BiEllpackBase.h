// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/VectorView.h>

#include "ElementsOrganization.h"
#include "BiEllpackSegmentView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
class BiEllpackBase
{
public:
   using DeviceType = Device;

   using IndexType = std::remove_const_t< Index >;

   using OffsetsView = Containers::VectorView< Index, DeviceType, IndexType >;

   using ConstOffsetsView = typename OffsetsView::ConstViewType;

   using SegmentViewType = BiEllpackSegmentView< IndexType, Organization, WarpSize >;

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
   BiEllpackBase() = default;

   __cuda_callable__
   BiEllpackBase( IndexType size, IndexType storageSize, OffsetsView segmentsPermutation, OffsetsView groupPointers );

   __cuda_callable__
   BiEllpackBase( const BiEllpackBase& ) = default;

   __cuda_callable__
   BiEllpackBase( BiEllpackBase&& ) noexcept = default;

   BiEllpackBase&
   operator=( const BiEllpackBase& ) = delete;

   BiEllpackBase&
   operator=( BiEllpackBase&& ) = delete;

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
   OffsetsView
   getSegmentsPermutationView();

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSegmentsPermutationView() const;

   [[nodiscard]] __cuda_callable__
   OffsetsView
   getGroupPointersView();

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getGroupPointersView() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getVirtualSegments() const;

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

   void
   printStructure( std::ostream& str ) const;

protected:
   IndexType size = 0;
   IndexType storageSize = 0;
   OffsetsView segmentsPermutation;
   OffsetsView groupPointers;

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
   bind( IndexType size, IndexType storageSize, OffsetsView segmentsPermutation, OffsetsView groupPointers );

   [[nodiscard]] __cuda_callable__
   IndexType
   getVirtualSegments( IndexType segmentsCount ) const;
};

}  // namespace TNL::Algorithms::Segments

#include "BiEllpackBase.hpp"
#include "print.h"
