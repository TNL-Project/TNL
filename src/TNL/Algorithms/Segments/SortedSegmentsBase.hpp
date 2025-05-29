// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Backend.h>
#include <TNL/TypeTraits.h>

#include "SortedSegmentsBase.h"

namespace TNL::Algorithms::Segments {

template< typename EmbeddedSegments, typename Index >
std::string
SortedSegmentsBase< EmbeddedSegments, Index >::getSerializationType()
{
   return "SortedSegments< " + EmbeddedSegments::getSerializationType() + " >";
}

template< typename EmbeddedSegments, typename Index >
std::string
SortedSegmentsBase< EmbeddedSegments, Index >::getSegmentsType()
{
   return "SortedSegments";
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getSegmentsCount() const -> IndexType
{
   return this->embeddedSegmentsView.getSegmentsCount();
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getSegmentSize( IndexType segmentIdx ) const -> IndexType
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

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getSize() const -> IndexType
{
   return this->embeddedSegmentsView.getSize();
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getStorageSize() const -> IndexType
{
   return this->embeddedSegmentsView.getStorageSize();
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getGlobalIndex( const IndexType segmentIdx, const IndexType localIdx ) const
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

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getSegmentView( IndexType segmentIdx ) const -> SegmentViewType
{
   return this->embeddedSegmentsView.getSegmentView( this->segmentsPermutationView[ segmentIdx ] );
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getEmbeddedSegmentsView() const -> EmbeddedSegmentsConstView
{
   return embeddedSegmentsView.getConstView();
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getEmbeddedSegmentsView() -> EmbeddedSegmentsView
{
   return embeddedSegmentsView.getView();
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getSegmentsPermutationView() -> PermutationView
{
   return segmentsPermutationView;
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getSegmentsPermutationView() const -> ConstPermutationView
{
   return segmentsPermutationView.getConstView();
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getInverseSegmentsPermutationView() -> PermutationView
{
   return inverseSegmentsPermutationView;
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsBase< EmbeddedSegments, Index >::getInverseSegmentsPermutationView() const -> ConstPermutationView
{
   return inverseSegmentsPermutationView.getConstView();
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
void
SortedSegmentsBase< EmbeddedSegments, Index >::bind( EmbeddedSegmentsView&& embeddedSegments,
                                                     PermutationView&& segmentsPermutation,
                                                     PermutationView&& inverseSegmentsPermutation )
{
   this->embeddedSegmentsView.bind( std::move( embeddedSegments ) );
   this->segmentsPermutationView.bind( std::move( segmentsPermutation ) );
   this->inverseSegmentsPermutationView.bind( std::move( inverseSegmentsPermutation ) );
}

}  // namespace TNL::Algorithms::Segments
