// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "CSRBase.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index >
__cuda_callable__
void
CSRBase< Device, Index >::bind( OffsetsView offsets )
{
   this->offsets.bind( std::move( offsets ) );
}

template< typename Device, typename Index >
__cuda_callable__
CSRBase< Device, Index >::CSRBase( const OffsetsView& offsets ) : offsets( offsets ) {}

template< typename Device, typename Index >
__cuda_callable__
CSRBase< Device, Index >::CSRBase( OffsetsView&& offsets ) : offsets( std::move( offsets ) ) {}

template< typename Device, typename Index >
std::string
CSRBase< Device, Index >::getSerializationType()
{
   return "CSR< " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device, typename Index >
String
CSRBase< Device, Index >::getSegmentsType()
{
   return "CSR";
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getSegmentsCount() const -> IndexType
{
   return this->offsets.getSize() - 1;
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getSegmentSize( IndexType segmentIdx ) const -> IndexType
{
   if( ! std::is_same< DeviceType, Devices::Host >::value ) {
#ifdef __CUDA_ARCH__
      return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
#else
      return offsets.getElement( segmentIdx + 1 ) - offsets.getElement( segmentIdx );
#endif
   }
   return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getSize() const -> IndexType
{
   return this->getStorageSize();
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getStorageSize() const -> IndexType
{
   if( ! std::is_same< DeviceType, Devices::Host >::value ) {
#ifdef __CUDA_ARCH__
      return offsets[ getSegmentsCount() ];
#else
      return offsets.getElement( getSegmentsCount() );
#endif
   }
   return offsets[ getSegmentsCount() ];
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
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
CSRBase< Device, Index >::getSegmentView( IndexType segmentIdx ) const -> SegmentViewType
{
   return SegmentViewType( segmentIdx, offsets[ segmentIdx ], offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ] );
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getOffsets() -> OffsetsView
{
   return this->offsets;
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getOffsets() const -> ConstOffsetsView
{
   return this->offsets.getConstView();
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::forElements( IndexType begin, IndexType end, Function&& function ) const
{
   const auto offsetsView = this->offsets;
   auto l = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      const IndexType begin = offsetsView[ segmentIdx ];
      const IndexType end = offsetsView[ segmentIdx + 1 ];
      IndexType localIdx( 0 );
      for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
         function( segmentIdx, localIdx++, globalIdx );
   };
   Algorithms::parallelFor< Device >( begin, end, l );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::forAllElements( Function&& function ) const
{
   this->forElements( 0, this->getSegmentsCount(), function );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   const auto& self = *this;
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = self.getSegmentView( segmentIdx );
      function( segment );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::forAllSegments( Function&& function ) const
{
   this->forSegments( 0, this->getSegmentsCount(), function );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::sequentialForSegments( IndexType begin, IndexType end, Function&& function ) const
{
   for( IndexType i = begin; i < end; i++ )
      forSegments( i, i + 1, function );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::sequentialForAllSegments( Function&& function ) const
{
   this->sequentialForSegments( 0, this->getSegmentsCount(), function );
}

}  // namespace TNL::Algorithms::Segments
