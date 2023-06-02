// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "CSR.h"
#include "detail/CSR.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, typename IndexAllocator >
template< typename SizesContainer >
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
std::string
CSR< Device, Index, IndexAllocator >::getSerializationType()
{
   return ViewType::getSerializationType();
}

template< typename Device, typename Index, typename IndexAllocator >
String
CSR< Device, Index, IndexAllocator >::getSegmentsType()
{
   return ViewType::getSegmentsType();
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename SizesHolder >
void
CSR< Device, Index, IndexAllocator >::setSegmentsSizes( const SizesHolder& sizes )
{
   detail::CSR< Device, Index >::setSegmentsSizes( sizes, this->offsets );
}

template< typename Device, typename Index, typename IndexAllocator >
void
CSR< Device, Index, IndexAllocator >::reset()
{
   this->offsets.setSize( 1 );
   this->offsets = 0;
}

template< typename Device, typename Index, typename IndexAllocator >
typename CSR< Device, Index, IndexAllocator >::ViewType
CSR< Device, Index, IndexAllocator >::getView()
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
__cuda_callable__
auto
CSR< Device, Index, IndexAllocator >::getSegmentsCount() const -> IndexType
{
   return this->offsets.getSize() - 1;
}

template< typename Device, typename Index, typename IndexAllocator >
__cuda_callable__
auto
CSR< Device, Index, IndexAllocator >::getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return detail::CSR< Device, Index >::getSegmentSize( this->offsets, segmentIdx );
}

template< typename Device, typename Index, typename IndexAllocator >
__cuda_callable__
auto
CSR< Device, Index, IndexAllocator >::getSize() const -> IndexType
{
   return this->getStorageSize();
}

template< typename Device, typename Index, typename IndexAllocator >
__cuda_callable__
auto
CSR< Device, Index, IndexAllocator >::getStorageSize() const -> IndexType
{
   return detail::CSR< Device, Index >::getStorageSize( this->offsets );
}

template< typename Device, typename Index, typename IndexAllocator >
__cuda_callable__
auto
CSR< Device, Index, IndexAllocator >::getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
{
   if( ! std::is_same< DeviceType, Devices::Host >::value ) {
#ifdef __CUDA_ARCH__
      return offsets[ segmentIdx ] + localIdx;
#else
      return offsets.getElement( segmentIdx ) + localIdx;
#endif
   }
   return offsets[ segmentIdx ] + localIdx;
}

template< typename Device, typename Index, typename IndexAllocator >
__cuda_callable__
auto
CSR< Device, Index, IndexAllocator >::getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   return SegmentViewType( segmentIdx, offsets[ segmentIdx ], offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ] );
}

template< typename Device, typename Index, typename IndexAllocator >
auto
CSR< Device, Index, IndexAllocator >::getOffsets() const -> const OffsetsContainer&
{
   return this->offsets;
}

template< typename Device, typename Index, typename IndexAllocator >
auto
CSR< Device, Index, IndexAllocator >::getOffsets() -> OffsetsContainer&
{
   return this->offsets;
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename Function >
void
CSR< Device, Index, IndexAllocator >::forElements( IndexType begin, IndexType end, Function&& f ) const
{
   this->getConstView().forElements( begin, end, f );
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename Function >
void
CSR< Device, Index, IndexAllocator >::forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename Function >
void
CSR< Device, Index, IndexAllocator >::forSegments( IndexType begin, IndexType end, Function&& f ) const
{
   this->getConstView().forSegments( begin, end, f );
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename Function >
void
CSR< Device, Index, IndexAllocator >::forAllSegments( Function&& f ) const
{
   this->getConstView().forAllSegments( f );
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename Function >
void
CSR< Device, Index, IndexAllocator >::sequentialForSegments( IndexType begin, IndexType end, Function&& f ) const
{
   this->getConstView().sequentialForSegments( begin, end, f );
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename Function >
void
CSR< Device, Index, IndexAllocator >::sequentialForAllSegments( Function&& f ) const
{
   this->getConstView().sequentialForAllSegments( f );
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename Device_, typename Index_, typename IndexAllocator_ >
CSR< Device, Index, IndexAllocator >&
CSR< Device, Index, IndexAllocator >::operator=( const CSR< Device_, Index_, IndexAllocator_ >& source )
{
   this->offsets = source.offsets;
   return *this;
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
}

}  // namespace TNL::Algorithms::Segments
