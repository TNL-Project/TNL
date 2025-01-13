// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "SlicedEllpackView.h"

namespace TNL::Algorithms::Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int SliceSize = 32 >
class SlicedEllpack : public SlicedEllpackBase< Device, Index, Organization, SliceSize >
{
   using Base = SlicedEllpackBase< Device, Index, Organization, SliceSize >;

public:
   using ViewType = SlicedEllpackView< Device, Index, Organization, SliceSize >;

   using ConstViewType = SlicedEllpackView< Device, std::add_const_t< Index >, Organization, SliceSize >;

   template< typename Device_, typename Index_ >
   using ViewTemplate = SlicedEllpackView< Device_, Index_, Organization, SliceSize >;

   using OffsetsContainer = Containers::Vector< Index, Device, typename Base::IndexType, IndexAllocator >;

   SlicedEllpack() = default;

   template< typename SizesContainer, typename T = std::enable_if_t< IsArrayType< SizesContainer >::value > >
   explicit SlicedEllpack( const SizesContainer& segmentsSizes );

   template< typename ListIndex >
   SlicedEllpack( const std::initializer_list< ListIndex >& segmentsSizes );

   SlicedEllpack( const SlicedEllpack& );

   SlicedEllpack( SlicedEllpack&& ) noexcept = default;

   //! \brief Copy-assignment operator (makes a deep copy).
   SlicedEllpack&
   operator=( const SlicedEllpack& segments );

   //! \brief Move-assignment operator.
   SlicedEllpack&
   operator=( SlicedEllpack&& ) noexcept( false );

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
   SlicedEllpack&
   operator=( const SlicedEllpack< Device_, Index_, IndexAllocator_, Organization_, SliceSize >& segments );

   [[nodiscard]] ViewType
   getView();

   [[nodiscard]] ConstViewType
   getConstView() const;

   /**
    * \brief Set sizes of particular segments.
    */
   template< typename SizesHolder = OffsetsContainer >
   void
   setSegmentsSizes( const SizesHolder& sizes );

   void
   reset();

   void
   save( File& file ) const;

   void
   load( File& file );

protected:
   OffsetsContainer sliceOffsets;
   OffsetsContainer sliceSegmentSizes;
};

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int SliceSize = 32 >
struct RowMajorSlicedEllpack : public SlicedEllpack< Device, Index, IndexAllocator, RowMajorOrder, SliceSize >
{
   using BaseType = SlicedEllpack< Device, Index, IndexAllocator, RowMajorOrder, SliceSize >;

   RowMajorSlicedEllpack() = default;

   template< typename SizesContainer, typename T = std::enable_if_t< IsArrayType< SizesContainer >::value > >
   explicit RowMajorSlicedEllpack( const SizesContainer& segmentsSizes ) : BaseType( segmentsSizes )
   {}

   template< typename ListIndex >
   RowMajorSlicedEllpack( const std::initializer_list< ListIndex >& segmentsSizes ) : BaseType( segmentsSizes )
   {}

   RowMajorSlicedEllpack( const RowMajorSlicedEllpack& );

   RowMajorSlicedEllpack( RowMajorSlicedEllpack&& ) noexcept = default;
};

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int SliceSize = 32 >
struct ColumnMajorSlicedEllpack : public SlicedEllpack< Device, Index, IndexAllocator, ColumnMajorOrder, SliceSize >
{
   using BaseType = SlicedEllpack< Device, Index, IndexAllocator, ColumnMajorOrder, SliceSize >;

   ColumnMajorSlicedEllpack() = default;

   template< typename SizesContainer, typename T = std::enable_if_t< IsArrayType< SizesContainer >::value > >
   explicit ColumnMajorSlicedEllpack( const SizesContainer& segmentsSizes ) : BaseType( segmentsSizes )
   {}

   template< typename ListIndex >
   ColumnMajorSlicedEllpack( const std::initializer_list< ListIndex >& segmentsSizes ) : BaseType( segmentsSizes )
   {}

   ColumnMajorSlicedEllpack( const ColumnMajorSlicedEllpack& );

   ColumnMajorSlicedEllpack( ColumnMajorSlicedEllpack&& ) noexcept = default;
};

}  // namespace TNL::Algorithms::Segments

#include "SlicedEllpack.hpp"
