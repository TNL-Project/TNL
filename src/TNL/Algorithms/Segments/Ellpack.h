// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "EllpackView.h"

namespace TNL::Algorithms::Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int Alignment = 32 >
class Ellpack : public EllpackBase< Device, Index, Organization, Alignment >
{
   using Base = EllpackBase< Device, Index, Organization, Alignment >;

public:
   using ViewType = EllpackView< Device, Index, Organization, Alignment >;

   using ConstViewType = typename ViewType::ConstViewType;

   template< typename Device_, typename Index_ >
   using ViewTemplate = EllpackView< Device_, Index_, Organization, Alignment >;

   using OffsetsContainer = Containers::Vector< Index, Device, typename Base::IndexType, IndexAllocator >;

   Ellpack() = default;

   template< typename SizesContainer >
   Ellpack( const SizesContainer& sizes );

   template< typename ListIndex >
   Ellpack( const std::initializer_list< ListIndex >& segmentsSizes );

   Ellpack( Index segmentsCount, Index segmentSize );

   Ellpack( const Ellpack& segments ) = default;

   Ellpack( Ellpack&& segments ) noexcept = default;

   //! \brief Copy-assignment operator.
   Ellpack&
   operator=( const Ellpack& segments );

   //! \brief Move-assignment operator.
   Ellpack&
   operator=( Ellpack&& ) noexcept;

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_, int Alignment_ >
   Ellpack&
   operator=( const Ellpack< Device_, Index_, IndexAllocator_, Organization_, Alignment_ >& segments );

   [[nodiscard]] ViewType
   getView();

   [[nodiscard]] ConstViewType
   getConstView() const;

   /**
    * \brief Set sizes of particular segments.
    */
   template< typename SizesContainer >
   void
   setSegmentsSizes( const SizesContainer& sizes );

   void
   setSegmentsSizes( Index segmentsCount, Index segmentSize );

   void
   reset();

   void
   save( File& file ) const;

   void
   load( File& file );
};

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int Alignment = 32 >
struct RowMajorEllpack : public Ellpack< Device, Index, IndexAllocator, RowMajorOrder, Alignment >
{
   using BaseType = Ellpack< Device, Index, IndexAllocator, RowMajorOrder, Alignment >;

   RowMajorEllpack() = default;

   template< typename SizesContainer >
   RowMajorEllpack( const SizesContainer& sizes ) : BaseType( sizes )
   {}

   template< typename ListIndex >
   RowMajorEllpack( const std::initializer_list< ListIndex >& segmentsSizes ) : BaseType( segmentsSizes )
   {}

   RowMajorEllpack( Index segmentsCount, Index segmentSize ) : BaseType( segmentsCount, segmentSize ) {}

   RowMajorEllpack( const RowMajorEllpack& segments ) = default;

   RowMajorEllpack( RowMajorEllpack&& segments ) noexcept = default;
};

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int Alignment = 32 >
struct ColumnMajorEllpack : public Ellpack< Device, Index, IndexAllocator, ColumnMajorOrder, Alignment >
{
   using BaseType = Ellpack< Device, Index, IndexAllocator, ColumnMajorOrder, Alignment >;

   ColumnMajorEllpack() = default;

   template< typename SizesContainer >
   ColumnMajorEllpack( const SizesContainer& sizes ) : BaseType( sizes )
   {}

   template< typename ListIndex >
   ColumnMajorEllpack( const std::initializer_list< ListIndex >& segmentsSizes ) : BaseType( segmentsSizes )
   {}

   ColumnMajorEllpack( Index segmentsCount, Index segmentSize ) : BaseType( segmentsCount, segmentSize ) {}

   ColumnMajorEllpack( const ColumnMajorEllpack& segments ) = default;

   ColumnMajorEllpack( ColumnMajorEllpack&& segments ) noexcept = default;
};

template< typename Segments >
struct isEllpackSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
struct isEllpackSegments< Ellpack< Device, Index, IndexAllocator, Organization, Alignment > > : std::true_type
{};

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
struct isEllpackSegments< EllpackView< Device, Index, Organization, Alignment > > : std::true_type
{};

template< typename Device, typename Index, typename IndexAllocator, int Alignment >
struct isEllpackSegments< ColumnMajorEllpack< Device, Index, IndexAllocator, Alignment > > : std::true_type
{};

template< typename Device, typename Index, int Alignment >
struct isEllpackSegments< ColumnMajorEllpackView< Device, Index, Alignment > > : std::true_type
{};

template< typename Device, typename Index, typename IndexAllocator, int Alignment >
struct isEllpackSegments< RowMajorEllpack< Device, Index, IndexAllocator, Alignment > > : std::true_type
{};
template< typename Device, typename Index, int Alignment >
struct isEllpackSegments< RowMajorEllpackView< Device, Index, Alignment > > : std::true_type
{};

template< typename Segments >
inline constexpr bool isEllpackSegments_v = isEllpackSegments< Segments >::value;

}  // namespace TNL::Algorithms::Segments

#include "Ellpack.hpp"
