// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SlicedEllpackBase.h"

namespace TNL::Algorithms::Segments {

template< typename Device,
          typename Index,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int SliceSize = 32 >
class SlicedEllpackView : public SlicedEllpackBase< Device, Index, Organization, SliceSize >
{
   using Base = SlicedEllpackBase< Device, Index, Organization, SliceSize >;

public:
   using ViewType = SlicedEllpackView;

   using ConstViewType = SlicedEllpackView< Device, std::add_const_t< Index >, Organization, SliceSize >;

   template< typename Device_, typename Index_ >
   using ViewTemplate = SlicedEllpackView< Device_, Index_, Organization, SliceSize >;

   __cuda_callable__
   SlicedEllpackView() = default;

   __cuda_callable__
   SlicedEllpackView( Index size,
                      Index alignedSize,
                      Index segmentsCount,
                      typename Base::OffsetsView sliceOffsets,
                      typename Base::OffsetsView sliceSegmentSizes );

   __cuda_callable__
   SlicedEllpackView( const SlicedEllpackView& ) = default;

   __cuda_callable__
   SlicedEllpackView( SlicedEllpackView&& ) noexcept = default;

   SlicedEllpackView&
   operator=( const SlicedEllpackView& ) = delete;

   SlicedEllpackView&
   operator=( SlicedEllpackView&& ) = delete;

   __cuda_callable__
   void
   bind( SlicedEllpackView view );

   [[nodiscard]] __cuda_callable__
   ViewType
   getView();

   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const;

   void
   save( File& file ) const;

   void
   load( File& file );
};

}  // namespace TNL::Algorithms::Segments

#include "SlicedEllpackView.hpp"
