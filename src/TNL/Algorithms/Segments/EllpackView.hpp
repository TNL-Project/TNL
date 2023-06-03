// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "EllpackView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::EllpackView( IndexType segmentsCount,
                                                                    IndexType segmentSize,
                                                                    IndexType alignedSize )
: segmentSize( segmentSize ), segmentsCount( segmentsCount ), alignedSize( alignedSize )
{}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
void
EllpackView< Device, Index, Organization, Alignment >::bind( EllpackView& view )
{
   this->segmentSize = view.segmentSize;
   this->segmentsCount = view.segmentsCount;
   this->alignedSize = view.alignedSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
void
EllpackView< Device, Index, Organization, Alignment >::bind( EllpackView&& view )
{
   this->segmentSize = view.segmentSize;
   this->segmentsCount = view.segmentsCount;
   this->alignedSize = view.alignedSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::EllpackView( IndexType segmentsCount, IndexType segmentSize )
: segmentSize( segmentSize ), segmentsCount( segmentsCount )
{
   if( Organization == RowMajorOrder )
      this->alignedSize = this->segmentsCount;
   else
      this->alignedSize = roundUpDivision( segmentsCount, this->getAlignment() ) * this->getAlignment();
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
std::string
EllpackView< Device, Index, Organization, Alignment >::getSerializationType()
{
   return "Ellpack< " + TNL::getSerializationType< IndexType >() + ", " + TNL::getSerializationType( Organization ) + ", "
        + std::to_string( Alignment ) + " >";
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
String
EllpackView< Device, Index, Organization, Alignment >::getSegmentsType()
{
   return "Ellpack";
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
typename EllpackView< Device, Index, Organization, Alignment >::ViewType
EllpackView< Device, Index, Organization, Alignment >::getView()
{
   return { segmentsCount, segmentSize, alignedSize };
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getConstView() const -> ConstViewType
{
   return { segmentsCount, segmentSize, alignedSize };
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getSegmentsCount() const -> IndexType
{
   return this->segmentsCount;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return this->segmentSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getSize() const -> IndexType
{
   return this->segmentsCount * this->segmentSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getStorageSize() const -> IndexType
{
   return this->alignedSize * this->segmentSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getGlobalIndex( const Index segmentIdx, const Index localIdx ) const
   -> IndexType
{
   if( Organization == RowMajorOrder )
      return segmentIdx * this->segmentSize + localIdx;
   else
      return segmentIdx + this->alignedSize * localIdx;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   if( Organization == RowMajorOrder )
      return SegmentViewType( segmentIdx, segmentIdx * this->segmentSize, this->segmentSize, 1 );
   else
      return SegmentViewType( segmentIdx, segmentIdx, this->segmentSize, this->alignedSize );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getAlignedSize() const -> IndexType
{
   return alignedSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
EllpackView< Device, Index, Organization, Alignment >::forElements( IndexType begin, IndexType end, Function&& f ) const
{
   if( Organization == RowMajorOrder ) {
      const IndexType segmentSize = this->segmentSize;
      auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         const IndexType begin = segmentIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         IndexType localIdx( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
            f( segmentIdx, localIdx++, globalIdx );
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }
   else {
      const IndexType storageSize = this->getStorageSize();
      const IndexType alignedSize = this->alignedSize;
      auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         const IndexType begin = segmentIdx;
         const IndexType end = storageSize;
         IndexType localIdx( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
            f( segmentIdx, localIdx++, globalIdx );
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
EllpackView< Device, Index, Organization, Alignment >::forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
EllpackView< Device, Index, Organization, Alignment >::forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   auto view = this->getConstView();
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = view.getSegmentView( segmentIdx );
      function( segment );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
EllpackView< Device, Index, Organization, Alignment >::forAllSegments( Function&& f ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
void
EllpackView< Device, Index, Organization, Alignment >::save( File& file ) const
{
   file.save( &segmentSize );
   file.save( &segmentsCount );
   file.save( &alignedSize );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
void
EllpackView< Device, Index, Organization, Alignment >::load( File& file )
{
   file.load( &segmentSize );
   file.load( &segmentsCount );
   file.load( &alignedSize );
}

}  // namespace TNL::Algorithms::Segments
