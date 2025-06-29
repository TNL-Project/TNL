// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Backend.h>
#include <TNL/TypeTraits.h>

#include "SortedSegmentsBase.h"

namespace TNL::Algorithms::Segments {

template< typename EmbeddedSegments >
std::string
SortedSegmentsBase< EmbeddedSegments >::getSerializationType()
{
   return "SortedSegments< " + EmbeddedSegments::getSerializationType() + " >";
}

template< typename EmbeddedSegments >
std::string
SortedSegmentsBase< EmbeddedSegments >::getSegmentsType()
{
   return "Sorted " + EmbeddedSegments::getSegmentsType();
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getSegmentsCount() const -> IndexType
{
   return this->embeddedSegmentsView.getSegmentsCount();
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getSegmentSize( IndexType segmentIdx ) const -> IndexType
{
   if( ! std::is_same_v< DeviceType, Devices::Host > && ! std::is_same_v< DeviceType, Devices::Sequential > ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return embeddedSegmentsView.getSegmentSize( segmentsPermutationView[ segmentIdx ] );
#else
      return embeddedSegmentsView.getSegmentSize( segmentsPermutationView.getElement( segmentIdx ) );
#endif
   }
   return embeddedSegmentsView.getSegmentSize( segmentsPermutationView[ segmentIdx ] );
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getSize() const -> IndexType
{
   return this->embeddedSegmentsView.getSize();
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getStorageSize() const -> IndexType
{
   return this->embeddedSegmentsView.getStorageSize();
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getGlobalIndex( const IndexType segmentIdx, const IndexType localIdx ) const
   -> IndexType
{
   if( ! std::is_same_v< DeviceType, Devices::Host > && ! std::is_same_v< DeviceType, Devices::Sequential > ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return this->embeddedSegmentsView.getGlobalIndex( this->segmentsPermutationView[ segmentIdx ], localIdx );
#else
      return this->embeddedSegmentsView.getGlobalIndex( this->segmentsPermutationView.getElement( segmentIdx ), localIdx );
#endif
   }
   return this->embeddedSegmentsView.getGlobalIndex( this->segmentsPermutationView[ segmentIdx ], localIdx );
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getSegmentView( IndexType segmentIdx ) const -> SegmentViewType
{
   return this->embeddedSegmentsView.getSegmentView( this->segmentsPermutationView[ segmentIdx ] );
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getEmbeddedSegmentsView() const -> EmbeddedSegmentsConstView
{
   return embeddedSegmentsView.getConstView();
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getEmbeddedSegmentsView() -> EmbeddedSegmentsView
{
   return embeddedSegmentsView.getView();
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getSegmentsPermutationView() -> PermutationView
{
   return segmentsPermutationView;
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getSegmentsPermutationView() const -> ConstPermutationView
{
   return segmentsPermutationView.getConstView();
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getInverseSegmentsPermutationView() -> PermutationView
{
   return inverseSegmentsPermutationView;
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments >::getInverseSegmentsPermutationView() const -> ConstPermutationView
{
   return inverseSegmentsPermutationView.getConstView();
}

template< typename EmbeddedSegments >
__cuda_callable__
void
SortedSegmentsBase< EmbeddedSegments >::bind( const EmbeddedSegmentsView& embeddedSegments,
                                              const PermutationView& segmentsPermutation,
                                              const PermutationView& inverseSegmentsPermutation )
{
   this->embeddedSegmentsView.bind( embeddedSegments );
   this->segmentsPermutationView.bind( segmentsPermutation );
   this->inverseSegmentsPermutationView.bind( inverseSegmentsPermutation );
}

}  // namespace TNL::Algorithms::Segments
