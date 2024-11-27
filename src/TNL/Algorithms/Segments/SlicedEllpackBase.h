// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/VectorView.h>

#include "ElementsOrganization.h"
#include "SegmentView.h"
#include "printSegments.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
class SlicedEllpackBase
{
public:
   using DeviceType = Device;

   using IndexType = std::remove_const_t< Index >;

   using OffsetsView = Containers::VectorView< Index, DeviceType, IndexType >;

   using ConstOffsetsView = typename OffsetsView::ConstViewType;

   using SegmentViewType = SegmentView< IndexType, Organization >;

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

   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return true;
   }

   __cuda_callable__
   SlicedEllpackBase() = default;

   __cuda_callable__
   SlicedEllpackBase( IndexType size,
                      IndexType storageSize,
                      IndexType segmentsCount,
                      OffsetsView&& sliceOffsets,
                      OffsetsView&& sliceSegmentSizes );

   __cuda_callable__
   SlicedEllpackBase( const SlicedEllpackBase& ) = default;

   __cuda_callable__
   SlicedEllpackBase( SlicedEllpackBase&& ) noexcept = default;

   SlicedEllpackBase&
   operator=( const SlicedEllpackBase& ) = delete;

   SlicedEllpackBase&
   operator=( SlicedEllpackBase&& ) = delete;

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
   getSliceSegmentSizesView();

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSliceSegmentSizesView() const;

   [[nodiscard]] __cuda_callable__
   OffsetsView
   getSliceOffsetsView();

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSliceOffsetsView() const;

   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function&& function ) const;

   template< typename Function >
   void
   forAllElements( Function&& function ) const;

   template< typename Array, typename Function >
   void
   forElements( const Array& segmentIndexes, Index begin, Index end, Function function ) const;

   template< typename Array, typename Function >
   void
   forElements( const Array& segmentIndexes, Function function ) const;

   template< typename Condition, typename Function >
   void
   forElementsIf( IndexType begin, IndexType end, Condition condition, Function function ) const;

   template< typename Condition, typename Function >
   void
   forAllElementsIf( Condition condition, Function function ) const;

   template< typename Function >
   void
   forSegments( IndexType begin, IndexType end, Function&& function ) const;

   template< typename Function >
   void
   forAllSegments( Function&& function ) const;

   // TODO: sequentialForSegments, sequentialForAllSegments

protected:
   IndexType size = 0;
   IndexType storageSize = 0;
   IndexType segmentsCount = 0;

   OffsetsView sliceOffsets;
   OffsetsView sliceSegmentSizes;

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
   bind( IndexType size,
         IndexType storageSize,
         IndexType segmentsCount,
         OffsetsView sliceOffsets,
         OffsetsView sliceSegmentSizes );
};

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
std::ostream&
operator<<( std::ostream& str, const SlicedEllpackBase< Device, Index, Organization, SliceSize >& segments )
{
   return printSegments( str, segments );
}

}  // namespace TNL::Algorithms::Segments

#include "SlicedEllpackBase.hpp"
