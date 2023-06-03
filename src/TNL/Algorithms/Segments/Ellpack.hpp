// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "Ellpack.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename SizesContainer >
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::Ellpack( const SizesContainer& segmentsSizes )
{
   setSegmentsSizes( segmentsSizes );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename ListIndex >
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::Ellpack(
   const std::initializer_list< ListIndex >& segmentsSizes )
{
   setSegmentsSizes( OffsetsContainer( segmentsSizes ) );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::Ellpack( const Index segmentsCount, const Index segmentSize )
{
   setSegmentsSizes( segmentsCount, segmentSize );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >&
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::operator=( const Ellpack& segments )
{
   Base::bind( segments.getSegmentsCount(), segments.getSegmentSize( 0 ), segments.getAlignedSize() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >&
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::operator=( Ellpack&& segments ) noexcept
{
   Base::bind( segments.getSegmentsCount(), segments.getSegmentSize( 0 ), segments.getAlignedSize() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_, int Alignment_ >
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >&
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::operator=(
   const Ellpack< Device_, Index_, IndexAllocator_, Organization_, Alignment_ >& segments )
{
   setSegmentsSizes( segments.getSegmentsCount(), segments.getSegmentSize( 0 ) );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
auto
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getView() -> ViewType
{
   return { this->segmentsCount, this->segmentSize, this->alignedSize };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
auto
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getConstView() const -> ConstViewType
{
   return { this->segmentsCount, this->segmentSize, this->alignedSize };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename SizesHolder >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::setSegmentsSizes( const SizesHolder& sizes )
{
   setSegmentsSizes( sizes.getSize(), max( sizes ) );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::setSegmentsSizes( const Index segmentsCount,
                                                                                     const Index segmentSize )
{
   if constexpr( Organization == RowMajorOrder )
      Base::bind( segmentsCount, segmentSize, segmentsCount );
   else
      Base::bind( segmentsCount, segmentSize, roundUpDivision( segmentsCount, this->getAlignment() ) * this->getAlignment() );
   if( integerMultiplyOverflow( this->alignedSize, this->segmentSize ) )
      throw( std::overflow_error( "Ellpack: multiplication overflow - the storage size required for the segments is larger "
                                  "than the maximal value of used index type." ) );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::reset()
{
   Base::bind( 0, 0, 0 );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::save( File& file ) const
{
   file.save( &this->segmentSize );
   file.save( &this->segmentsCount );
   file.save( &this->alignedSize );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::load( File& file )
{
   file.load( &this->segmentSize );
   file.load( &this->segmentsCount );
   file.load( &this->alignedSize );
}

}  // namespace TNL::Algorithms::Segments
