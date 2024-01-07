// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "EllpackBase.h"

namespace TNL::Algorithms::Segments {

template< typename Device,
          typename Index,
          ElementsOrganization Organization = Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int Alignment = 32 >
class EllpackView : public EllpackBase< Device, Index, Organization, Alignment >
{
   using Base = EllpackBase< Device, Index, Organization, Alignment >;

public:
   using ViewType = EllpackView;

   using ConstViewType = ViewType;

   template< typename Device_, typename Index_ >
   using ViewTemplate = EllpackView< Device_, Index_, Organization, Alignment >;

   __cuda_callable__
   EllpackView() = default;

   __cuda_callable__
   EllpackView( Index segmentsCount, Index segmentSize, Index alignedSize );

   __cuda_callable__
   EllpackView( Index segmentsCount, Index segmentSize );

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
   bind( EllpackView view );

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

#include "EllpackView.hpp"
