// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "CSRView.h"
#include "detail/CSR.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index >
__cuda_callable__
CSRView< Device, Index >::CSRView( const OffsetsView& offsets ) : offsets( offsets ) {}

template< typename Device, typename Index >
__cuda_callable__
CSRView< Device, Index >::CSRView( OffsetsView&& offsets ) : offsets( std::move( offsets ) ) {}

template< typename Device, typename Index >
__cuda_callable__
void
CSRView< Device, Index >::bind( CSRView& view )
{
   this->offsets.bind( view.offsets );
}

template< typename Device, typename Index >
__cuda_callable__
void
CSRView< Device, Index >::bind( CSRView&& view )
{
   this->offsets.bind( view.offsets );
}

template< typename Device, typename Index >
std::string
CSRView< Device, Index >::getSerializationType()
{
   return "CSR< " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device, typename Index >
String
CSRView< Device, Index >::getSegmentsType()
{
   return "CSR";
}

template< typename Device, typename Index >
__cuda_callable__
typename CSRView< Device, Index >::ViewType
CSRView< Device, Index >::getView()
{
   return { this->offsets };
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRView< Device, Index >::getConstView() const -> ConstViewType
{
   return { this->offsets.getConstView() };
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRView< Device, Index >::getSegmentsCount() const -> IndexType
{
   return this->offsets.getSize() - 1;
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRView< Device, Index >::getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return detail::CSR< Device, Index >::getSegmentSize( this->offsets, segmentIdx );
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRView< Device, Index >::getSize() const -> IndexType
{
   return this->getStorageSize();
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRView< Device, Index >::getStorageSize() const -> IndexType
{
   return detail::CSR< Device, Index >::getStorageSize( this->offsets );
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRView< Device, Index >::getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
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

template< typename Device, typename Index >
__cuda_callable__
auto
CSRView< Device, Index >::getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   return SegmentViewType( segmentIdx, offsets[ segmentIdx ], offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ], 1 );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRView< Device, Index >::forElements( IndexType begin, IndexType end, Function&& f ) const
{
   const auto offsetsView = this->offsets;
   auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
   {
      const IndexType begin = offsetsView[ segmentIdx ];
      const IndexType end = offsetsView[ segmentIdx + 1 ];
      IndexType localIdx( 0 );
      for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
         f( segmentIdx, localIdx++, globalIdx );
   };
   Algorithms::parallelFor< Device >( begin, end, l );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRView< Device, Index >::forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRView< Device, Index >::forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   auto view = this->getConstView();
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = view.getSegmentView( segmentIdx );
      function( segment );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRView< Device, Index >::forAllSegments( Function&& f ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRView< Device, Index >::sequentialForSegments( IndexType begin, IndexType end, Function&& function ) const
{
   for( IndexType i = begin; i < end; i++ )
      forSegments( i, i + 1, function );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRView< Device, Index >::sequentialForAllSegments( Function&& f ) const
{
   this->sequentialForSegments( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index >
void
CSRView< Device, Index >::save( File& file ) const
{
   file << this->offsets;
}

template< typename Device, typename Index >
void
CSRView< Device, Index >::load( File& file )
{
   file >> this->offsets;
}

}  // namespace TNL::Algorithms::Segments
