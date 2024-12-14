// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "BiEllpackView.h"

namespace TNL::Algorithms::Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int WarpSize = Backend::getWarpSize() >
class BiEllpack : public BiEllpackBase< Device, Index, Organization, WarpSize >
{
   using Base = BiEllpackBase< Device, Index, Organization, WarpSize >;

public:
   using ViewType = BiEllpackView< Device, Index, Organization, WarpSize >;

   using ConstViewType = typename ViewType::ConstViewType;

   template< typename Device_, typename Index_ >
   using ViewTemplate = BiEllpackView< Device_, Index_, Organization, WarpSize >;

   using OffsetsContainer = Containers::Vector< Index, Device, typename Base::IndexType, IndexAllocator >;

   BiEllpack() = default;

   template< typename SizesContainer >
   BiEllpack( const SizesContainer& segmentsSizes );

   template< typename ListIndex >
   BiEllpack( const std::initializer_list< ListIndex >& segmentsSizes );

   BiEllpack( const BiEllpack& segments );

   BiEllpack( BiEllpack&& segments ) noexcept = default;

   BiEllpack&
   operator=( const BiEllpack& segments );

   BiEllpack&
   operator=( BiEllpack&& segments ) noexcept( false );

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
   BiEllpack&
   operator=( const BiEllpack< Device_, Index_, IndexAllocator_, Organization_, WarpSize >& segments );

   [[nodiscard]] ViewType
   getView();

   [[nodiscard]] ConstViewType
   getConstView() const;

   template< typename SizesHolder >
   void
   setSegmentsSizes( const SizesHolder& sizes );

   void
   reset();

   void
   save( File& file ) const;

   void
   load( File& file );

   // This method is public only because of lambda functions in CUDA
   template< typename SizesHolder >
   void
   initGroupPointers( const SizesHolder& segmentsSizes );

protected:
   OffsetsContainer segmentsPermutation;
   OffsetsContainer groupPointers;

   template< typename SizesHolder >
   void
   initSegmentsPermutation( const SizesHolder& segmentsSize );

   template< typename SizesHolder >
   void
   verifySegmentPerm( const SizesHolder& segmentsSizes );

   template< typename SizesHolder >
   void
   verifySegmentLengths( const SizesHolder& segmentsSizes );

   [[nodiscard]] Index
   getStripLength( Index strip ) const;
};

}  // namespace TNL::Algorithms::Segments

#include "BiEllpack.hpp"
