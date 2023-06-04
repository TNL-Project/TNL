// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "SlicedEllpackBase.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
void
SlicedEllpackBase< Device, Index, Organization, SliceSize >::bind( IndexType size,
                                                                   IndexType storageSize,
                                                                   IndexType segmentsCount,
                                                                   OffsetsView sliceOffsets,
                                                                   OffsetsView sliceSegmentSizes )
{
   this->size = size;
   this->storageSize = storageSize;
   this->segmentsCount = segmentsCount;
   this->sliceOffsets.bind( std::move( sliceOffsets ) );
   this->sliceSegmentSizes.bind( std::move( sliceSegmentSizes ) );
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
SlicedEllpackBase< Device, Index, Organization, SliceSize >::SlicedEllpackBase( IndexType size,
                                                                                IndexType storageSize,
                                                                                IndexType segmentsCount,
                                                                                OffsetsView&& sliceOffsets,
                                                                                OffsetsView&& sliceSegmentSizes )
: size( size ), storageSize( storageSize ), segmentsCount( segmentsCount ), sliceOffsets( std::move( sliceOffsets ) ),
  sliceSegmentSizes( std::move( sliceSegmentSizes ) )
{}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
std::string
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getSerializationType()
{
   return "SlicedEllpack< " + TNL::getSerializationType< IndexType >() + ", " + TNL::getSerializationType( Organization ) + ", "
        + std::to_string( SliceSize ) + " >";
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
String
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getSegmentsType()
{
   return "SlicedEllpack";
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getSegmentsCount() const -> IndexType
{
   return this->segmentsCount;
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   const Index sliceIdx = segmentIdx / SliceSize;
   if constexpr( std::is_same< DeviceType, Devices::Host >::value )
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
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getSize() const -> IndexType
{
   return this->size;
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getGlobalIndex( const Index segmentIdx,
                                                                             const Index localIdx ) const -> IndexType
{
   const IndexType sliceIdx = segmentIdx / SliceSize;
   const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
   IndexType sliceOffset;
   IndexType segmentSize;
   if constexpr( std::is_same< DeviceType, Devices::Host >::value ) {
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
   if constexpr( Organization == RowMajorOrder )
      return sliceOffset + segmentInSliceIdx * segmentSize + localIdx;
   else
      return sliceOffset + segmentInSliceIdx + SliceSize * localIdx;
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getSegmentView( const IndexType segmentIdx ) const
   -> SegmentViewType
{
   const IndexType sliceIdx = segmentIdx / SliceSize;
   const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
   const IndexType& sliceOffset = this->sliceOffsets[ sliceIdx ];
   const IndexType& segmentSize = this->sliceSegmentSizes[ sliceIdx ];

   if constexpr( Organization == RowMajorOrder )
      return SegmentViewType( segmentIdx, sliceOffset + segmentInSliceIdx * segmentSize, segmentSize );
   else
      return SegmentViewType( segmentIdx, sliceOffset + segmentInSliceIdx, segmentSize, SliceSize );
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getSliceSegmentSizesView() -> OffsetsView
{
   return sliceSegmentSizes.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getSliceSegmentSizesView() const -> ConstOffsetsView
{
   return sliceSegmentSizes.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getSliceOffsetsView() -> OffsetsView
{
   return sliceOffsets.getView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackBase< Device, Index, Organization, SliceSize >::getSliceOffsetsView() const -> ConstOffsetsView
{
   return sliceOffsets.getConstView();
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
template< typename Function >
void
SlicedEllpackBase< Device, Index, Organization, SliceSize >::forElements( IndexType begin,
                                                                          IndexType end,
                                                                          Function&& function ) const
{
   const auto sliceSegmentSizes_view = this->sliceSegmentSizes.getConstView();
   const auto sliceOffsets_view = this->sliceOffsets.getConstView();
   if constexpr( Organization == RowMajorOrder ) {
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
            function( segmentIdx, localIdx++, globalIdx );
#endif
         }
      };
      Algorithms::parallelFor< Device >( begin, end, l );
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
            function( segmentIdx, localIdx++, globalIdx );
#endif
         }
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
template< typename Function >
void
SlicedEllpackBase< Device, Index, Organization, SliceSize >::forAllElements( Function&& function ) const
{
   this->forElements( 0, this->getSegmentsCount(), function );
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
template< typename Function >
void
SlicedEllpackBase< Device, Index, Organization, SliceSize >::forSegments( IndexType begin,
                                                                          IndexType end,
                                                                          Function&& function ) const
{
   const auto& self = *this;
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = self.getSegmentView( segmentIdx );
      function( segment );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
template< typename Function >
void
SlicedEllpackBase< Device, Index, Organization, SliceSize >::forAllSegments( Function&& function ) const
{
   this->forSegments( 0, this->getSegmentsCount(), function );
}

}  // namespace TNL::Algorithms::Segments
