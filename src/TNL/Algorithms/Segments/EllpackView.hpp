// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "EllpackView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::EllpackView( Index segmentsCount, Index segmentSize, Index alignedSize )
: Base( segmentsCount, segmentSize, alignedSize )
{}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::EllpackView( Index segmentsCount, Index segmentSize )
{
   // TODO: use concepts in C++20 to specify the requirement
   static_assert( Organization == Segments::RowMajorOrder || Alignment == 1 );
   Base::bind( segmentsCount, segmentSize, segmentsCount );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
void
EllpackView< Device, Index, Organization, Alignment >::bind( EllpackView view )
{
   Base::bind( view.segmentsCount, view.segmentSize, view.alignedSize );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
typename EllpackView< Device, Index, Organization, Alignment >::ViewType
EllpackView< Device, Index, Organization, Alignment >::getView()
{
   return { this->segmentsCount, this->segmentSize, this->alignedSize };
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getConstView() const -> ConstViewType
{
   return { this->segmentsCount, this->segmentSize, this->alignedSize };
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
void
EllpackView< Device, Index, Organization, Alignment >::save( File& file ) const
{
   file.save( &this->segmentSize );
   file.save( &this->segmentsCount );
   file.save( &this->alignedSize );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
void
EllpackView< Device, Index, Organization, Alignment >::load( File& file )
{
   file.load( &this->segmentSize );
   file.load( &this->segmentsCount );
   file.load( &this->alignedSize );
}

}  // namespace TNL::Algorithms::Segments
