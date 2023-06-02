// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "SlicedEllpackView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
SlicedEllpackView< Device, Index, Organization, SliceSize >::SlicedEllpackView( IndexType size,
                                                                                IndexType alignedSize,
                                                                                IndexType segmentsCount,
                                                                                OffsetsView&& sliceOffsets,
                                                                                OffsetsView&& sliceSegmentSizes )
: size( size ), alignedSize( alignedSize ), segmentsCount( segmentsCount ),
  sliceOffsets( std::forward< OffsetsView >( sliceOffsets ) ),
  sliceSegmentSizes( std::forward< OffsetsView >( sliceSegmentSizes ) )
{}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
std::string
SlicedEllpackView< Device, Index, Organization, SliceSize >::getSerializationType()
{
   return "SlicedEllpack< " + TNL::getSerializationType< IndexType >() + ", " + TNL::getSerializationType( Organization ) + ", "
        + std::to_string( SliceSize ) + " >";
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
String
SlicedEllpackView< Device, Index, Organization, SliceSize >::getSegmentsType()
{
   return "SlicedEllpack";
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
typename SlicedEllpackView< Device, Index, Organization, SliceSize >::ViewType
SlicedEllpackView< Device, Index, Organization, SliceSize >::getView()
{
   return { size, alignedSize, segmentsCount, sliceOffsets, sliceSegmentSizes };
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::getConstView() const -> ConstViewType
{
   return { size, alignedSize, segmentsCount, sliceOffsets.getConstView(), sliceSegmentSizes.getConstView() };
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::getSegmentsCount() const -> IndexType
{
   return this->segmentsCount;
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   const Index sliceIdx = segmentIdx / SliceSize;
   if( std::is_same< DeviceType, Devices::Host >::value )
      return this->sliceSegmentSizes[ sliceIdx ];
   else {
#ifdef __CUDA_ARCH__
      return this->sliceSegmentSizes[ sliceIdx ];
#else
      return this->sliceSegmentSizes.getElement( sliceIdx );
#endif
   }
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::getSize() const -> IndexType
{
   return this->size;
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::getStorageSize() const -> IndexType
{
   return this->alignedSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::getGlobalIndex( const Index segmentIdx,
                                                                             const Index localIdx ) const -> IndexType
{
   const IndexType sliceIdx = segmentIdx / SliceSize;
   const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
   IndexType sliceOffset;
   IndexType segmentSize;
   if( std::is_same< DeviceType, Devices::Host >::value ) {
      sliceOffset = this->sliceOffsets[ sliceIdx ];
      segmentSize = this->sliceSegmentSizes[ sliceIdx ];
   }
   else {
#ifdef __CUDA_ARCH__
      sliceOffset = this->sliceOffsets[ sliceIdx ];
      segmentSize = this->sliceSegmentSizes[ sliceIdx ];
#else
      sliceOffset = this->sliceOffsets.getElement( sliceIdx );
      segmentSize = this->sliceSegmentSizes.getElement( sliceIdx );
#endif
   }
   if( Organization == RowMajorOrder )
      return sliceOffset + segmentInSliceIdx * segmentSize + localIdx;
   else
      return sliceOffset + segmentInSliceIdx + SliceSize * localIdx;
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::getSegmentView( const IndexType segmentIdx ) const
   -> SegmentViewType
{
   const IndexType sliceIdx = segmentIdx / SliceSize;
   const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
   const IndexType& sliceOffset = this->sliceOffsets[ sliceIdx ];
   const IndexType& segmentSize = this->sliceSegmentSizes[ sliceIdx ];

   if( Organization == RowMajorOrder )
      return SegmentViewType( segmentIdx, sliceOffset + segmentInSliceIdx * segmentSize, segmentSize, 1 );
   else
      return SegmentViewType( segmentIdx, sliceOffset + segmentInSliceIdx, segmentSize, SliceSize );
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::getSliceSegmentSizesView() const -> ConstOffsetsView
{
   return sliceSegmentSizes.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::getSliceOffsetsView() const -> ConstOffsetsView
{
   return sliceOffsets.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
template< typename Function >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::forElements( IndexType first, IndexType last, Function&& f ) const
{
   const auto sliceSegmentSizes_view = this->sliceSegmentSizes.getConstView();
   const auto sliceOffsets_view = this->sliceOffsets.getConstView();
   if( Organization == RowMajorOrder ) {
      auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
         const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         IndexType localIdx( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
            // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
            f( segmentIdx, localIdx, globalIdx );
            localIdx++;
#else
            f( segmentIdx, localIdx++, globalIdx );
#endif
         }
      };
      Algorithms::parallelFor< Device >( first, last, l );
   }
   else {
      auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         // const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
         const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
         const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
         IndexType localIdx( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize ) {
            // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
            f( segmentIdx, localIdx, globalIdx );
            localIdx++;
#else
            f( segmentIdx, localIdx++, globalIdx );
#endif
         }
      };
      Algorithms::parallelFor< Device >( first, last, l );
   }
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
template< typename Function >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
template< typename Function >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::forSegments( IndexType begin,
                                                                          IndexType end,
                                                                          Function&& function ) const
{
   auto view = this->getConstView();
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = view.getSegmentView( segmentIdx );
      function( segment );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
template< typename Function >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::forAllSegments( Function&& f ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
SlicedEllpackView< Device, Index, Organization, SliceSize >&
SlicedEllpackView< Device, Index, Organization, SliceSize >::operator=(
   const SlicedEllpackView< Device, Index, Organization, SliceSize >& view )
{
   this->size = view.size;
   this->alignedSize = view.alignedSize;
   this->segmentsCount = view.segmentsCount;
   this->sliceOffsets.bind( view.sliceOffsets );
   this->sliceSegmentSizes.bind( view.sliceSegmentSizes );
   return *this;
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::save( File& file ) const
{
   file.save( &size );
   file.save( &alignedSize );
   file.save( &segmentsCount );
   file << this->sliceOffsets;
   file << this->sliceSegmentSizes;
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::load( File& file )
{
   file.load( &size );
   file.load( &alignedSize );
   file.load( &segmentsCount );
   file >> this->sliceOffsets;
   file >> this->sliceSegmentSizes;
}

}  // namespace TNL::Algorithms::Segments
