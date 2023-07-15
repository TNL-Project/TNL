// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "ChunkedEllpackView.h"

namespace TNL::Algorithms::Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
class ChunkedEllpack : public ChunkedEllpackBase< Device, Index, Organization >
{
   using Base = ChunkedEllpackBase< Device, Index, Organization >;

public:
   using ViewType = ChunkedEllpackView< Device, Index, Organization >;

   using ConstViewType = typename ViewType::ConstViewType;

   template< typename Device_, typename Index_ >
   using ViewTemplate = ChunkedEllpackView< Device_, Index_, Organization >;

   using OffsetsContainer = Containers::Vector< Index, Device, typename Base::IndexType, IndexAllocator >;

   using SliceInfoAllocator = typename Allocators::Default< Device >::template Allocator< typename Base::SliceInfoType >;
   using SliceInfoContainer =
      Containers::Array< typename TNL::copy_const< typename Base::SliceInfoType >::template from< Index >::type,
                         Device,
                         Index,
                         SliceInfoAllocator >;

   ChunkedEllpack() = default;

   template< typename SizesContainer >
   ChunkedEllpack( const SizesContainer& segmentsSizes );

   template< typename ListIndex >
   ChunkedEllpack( const std::initializer_list< ListIndex >& segmentsSizes );

   ChunkedEllpack( const ChunkedEllpack& segments );

   ChunkedEllpack( ChunkedEllpack&& segments ) noexcept = default;

   //! \brief Copy-assignment operator (makes a deep copy).
   ChunkedEllpack&
   operator=( const ChunkedEllpack& segments );

   //! \brief Move-assignment operator.
   ChunkedEllpack&
   operator=( ChunkedEllpack&& ) noexcept( false );

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
   ChunkedEllpack&
   operator=( const ChunkedEllpack< Device_, Index_, IndexAllocator_, Organization_ >& segments );

   [[nodiscard]] ViewType
   getView();

   [[nodiscard]] ConstViewType
   getConstView() const;

   template< typename SizesContainer >
   void
   setSegmentsSizes( const SizesContainer& segmentsSizes );

   void
   reset();

   void
   save( File& file ) const;

   void
   load( File& file );

protected:
   template< typename SizesContainer >
   void
   resolveSliceSizes( SizesContainer& segmentsSizes );

   template< typename SizesContainer >
   bool
   setSlice( SizesContainer& segmentsSizes, Index sliceIndex, Index& elementsToAllocation );

   //! \brief For each row, this keeps index of the first chunk within a slice.
   OffsetsContainer rowToChunkMapping;

   //! \brief For each segment, this keeps index of the slice which contains the segment.
   OffsetsContainer rowToSliceMapping;

   OffsetsContainer chunksToSegmentsMapping;

   //! \brief Keeps index of the first segment index.
   OffsetsContainer rowPointers;

   SliceInfoContainer slices;
};

}  // namespace TNL::Algorithms::Segments

#include "ChunkedEllpack.hpp"
