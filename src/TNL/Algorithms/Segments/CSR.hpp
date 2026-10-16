// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <stdexcept>

#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/DiscreteMath.h>

#include "CSR.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, typename IndexAllocator >
CSR< Device, Index, IndexAllocator >::CSR()
: offsets( 1 )
{
   this->offsets.setElement( 0, 0 );
   Base::bind( this->offsets.getView() );
}

template< typename Device, typename Index, typename IndexAllocator >
CSR< Device, Index, IndexAllocator >::CSR( const CSR& segments )
: offsets( segments.offsets )
{
   // update the base
   Base::bind( this->offsets.getView() );
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename SizesContainer, std::enable_if_t< IsArrayType< SizesContainer >::value, bool > >
CSR< Device, Index, IndexAllocator >::CSR( const SizesContainer& segmentsSizes )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename ListIndex >
CSR< Device, Index, IndexAllocator >::CSR( const std::initializer_list< ListIndex >& segmentsSizes )
{
   this->setSegmentsSizes( OffsetsContainer( segmentsSizes ) );
}

template< typename Device, typename Index, typename IndexAllocator >
CSR< Device, Index, IndexAllocator >&
CSR< Device, Index, IndexAllocator >::operator=( const CSR& segments )
{
   this->offsets = segments.offsets;
   // update the base
   Base::bind( this->offsets.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator >
CSR< Device, Index, IndexAllocator >&
CSR< Device, Index, IndexAllocator >::operator=( CSR&& segments ) noexcept( false )
{
   this->offsets = std::move( segments.offsets );
   // update the base
   Base::bind( this->offsets.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename Device_, typename Index_, typename IndexAllocator_ >
CSR< Device, Index, IndexAllocator >&
CSR< Device, Index, IndexAllocator >::operator=( const CSR< Device_, Index_, IndexAllocator_ >& segments )
{
   this->offsets = segments.getOffsets();
   // update the base
   Base::bind( this->offsets.getView() );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator >
auto
CSR< Device, Index, IndexAllocator >::getView() -> ViewType
{
   return { this->offsets.getView() };
}

template< typename Device, typename Index, typename IndexAllocator >
auto
CSR< Device, Index, IndexAllocator >::getConstView() const -> ConstViewType
{
   return { this->offsets.getConstView() };
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename SizesHolder >
void
CSR< Device, Index, IndexAllocator >::setSegmentsSizes( const SizesHolder& sizes )
{
   offsets.setSize( sizes.getSize() + 1 );
   // GOTCHA: when sizes.getSize() == 0, getView returns a full view with size == 1
   if( sizes.getSize() > 0 ) {
      auto view = offsets.getView( 0, sizes.getSize() );
      view = sizes;
   }
   offsets.setElement( sizes.getSize(), 0 );

   // The sum of the segments sizes saturates at the maximal value of the index
   // type, so it equals the maximal value if the scan would overflow.
   auto saturatingPlus = [] __cuda_callable__( Index a, Index b ) -> Index
   {
      return saturatingAdd( a, b );
   };
   if( Algorithms::reduce( offsets, saturatingPlus, static_cast< Index >( 0 ) ) == std::numeric_limits< Index >::max() )
      throw( std::overflow_error(
         "CSR: overflow - the storage size required for the segments is larger "
         "than the maximal value of used index type." ) );
   inplaceExclusiveScan( offsets );

   // update the base
   Base::bind( this->offsets.getView() );
}

template< typename Device, typename Index, typename IndexAllocator >
void
CSR< Device, Index, IndexAllocator >::reset()
{
   this->offsets.setSize( 1 );
   this->offsets = 0;

   // update the base
   Base::bind( this->offsets.getView() );
}

template< typename Device, typename Index, typename IndexAllocator >
void
CSR< Device, Index, IndexAllocator >::save( File& file ) const
{
   file << this->offsets;
}

template< typename Device, typename Index, typename IndexAllocator >
void
CSR< Device, Index, IndexAllocator >::load( File& file )
{
   file >> this->offsets;

   // update the base
   Base::bind( this->offsets.getView() );
}

}  // namespace TNL::Algorithms::Segments
