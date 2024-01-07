// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "BiEllpackBase.h"

namespace TNL::Algorithms::Segments {

template< typename Device,
          typename Index,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int WarpSize = 32 >
class BiEllpackView : public BiEllpackBase< Device, Index, Organization, WarpSize >
{
   using Base = BiEllpackBase< Device, Index, Organization, WarpSize >;

public:
   using ViewType = BiEllpackView;

   using ConstViewType = BiEllpackView< Device, std::add_const_t< Index >, Organization, WarpSize >;

   template< typename Device_, typename Index_ >
   using ViewTemplate = BiEllpackView< Device_, Index_, Organization, WarpSize >;

   __cuda_callable__
   BiEllpackView() = default;

   __cuda_callable__
   BiEllpackView( Index size,
                  Index storageSize,
                  typename Base::OffsetsView rowsPermutation,
                  typename Base::OffsetsView groupPointers );

   __cuda_callable__
   BiEllpackView( const BiEllpackView& ) = default;

   __cuda_callable__
   BiEllpackView( BiEllpackView&& ) noexcept = default;

   BiEllpackView&
   operator=( const BiEllpackView& ) = delete;

   BiEllpackView&
   operator=( BiEllpackView&& ) = delete;

   __cuda_callable__
   void
   bind( BiEllpackView view );

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

#include "BiEllpackView.hpp"
