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

template< typename Device, typename Index, int Alignment = 32 >
struct RowMajorEllpackView : public EllpackView< Device, Index, RowMajorOrder, Alignment >
{
   using Base = EllpackView< Device, Index, RowMajorOrder, Alignment >;

   __cuda_callable__
   RowMajorEllpackView() = default;

   __cuda_callable__
   RowMajorEllpackView( Index segmentsCount, Index segmentSize, Index alignedSize )
   : Base( segmentsCount, segmentSize, alignedSize )
   {}

   __cuda_callable__
   RowMajorEllpackView( Index segmentsCount, Index segmentSize ) : Base( segmentsCount, segmentSize ) {}

   __cuda_callable__
   RowMajorEllpackView( const RowMajorEllpackView& ) = default;

   __cuda_callable__
   RowMajorEllpackView( RowMajorEllpackView&& ) noexcept = default;
};

template< typename Device, typename Index, int Alignment = 32 >
struct ColumnMajorEllpackView : public EllpackView< Device, Index, ColumnMajorOrder, Alignment >
{
   using Base = EllpackView< Device, Index, ColumnMajorOrder, Alignment >;

   __cuda_callable__
   ColumnMajorEllpackView() = default;

   __cuda_callable__
   ColumnMajorEllpackView( Index segmentsCount, Index segmentSize, Index alignedSize )
   : Base( segmentsCount, segmentSize, alignedSize )
   {}

   __cuda_callable__
   ColumnMajorEllpackView( Index segmentsCount, Index segmentSize ) : Base( segmentsCount, segmentSize ) {}

   __cuda_callable__
   ColumnMajorEllpackView( const ColumnMajorEllpackView& ) = default;

   __cuda_callable__
   ColumnMajorEllpackView( ColumnMajorEllpackView&& ) noexcept = default;
};

}  // namespace TNL::Algorithms::Segments

#include "EllpackView.hpp"
