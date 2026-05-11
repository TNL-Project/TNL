// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SortedSegmentsView.h"

namespace TNL::Algorithms::Segments {

template< typename EmbeddedSegments >
__cuda_callable__
SortedSegmentsView< EmbeddedSegments >::SortedSegmentsView( EmbeddedSegmentsView embeddedSegmentsView,
                                                            PermutationView segmentsPermutation,
                                                            PermutationView inverseSegmentsPermutation )
{
   Base::bind( embeddedSegmentsView, segmentsPermutation, inverseSegmentsPermutation );
}

template< typename EmbeddedSegments >
__cuda_callable__
void
SortedSegmentsView< EmbeddedSegments >::bind( SortedSegmentsView&& view )
{
   this->embeddedSegmentsView.bind( view.getEmbeddedSegmentsView() );
   this->segmentsPermutationView.bind( view.getSegmentsPermutationView() );
   this->inverseSegmentsPermutationView.bind( view.getInverseSegmentsPermutationView() );
}

template< typename EmbeddedSegments >
void
SortedSegmentsView< EmbeddedSegments >::save( File& file ) const
{
   this->embeddedSegmentsView.save( file );
   file << this->segmentsPermutationView << this->inverseSegmentsPermutationView;
}

template< typename EmbeddedSegments >
void
SortedSegmentsView< EmbeddedSegments >::load( File& file )
{
   this->embeddedSegmentsView.load( file );
   file >> this->segmentsPermutationView >> this->inverseSegmentsPermutationView;
}

template< typename EmbeddedSegments >
__cuda_callable__
void
SortedSegmentsView< EmbeddedSegments >::bind( const EmbeddedSegmentsView& embeddedSegmentsView,
                                              const PermutationView& segmentsPermutation,
                                              const PermutationView& inverseSegmentsPermutation )
{
   this->embeddedSegmentsView.bind( embeddedSegmentsView );
   this->segmentsPermutationView.bind( segmentsPermutation );
   this->inverseSegmentsPermutationView.bind( inverseSegmentsPermutation );
}

template< typename EmbeddedSegments >
__cuda_callable__
void
SortedSegmentsView< EmbeddedSegments >::bind( const EmbeddedSegmentsConstView& embeddedSegmentsView,
                                              const ConstPermutationView& segmentsPermutation,
                                              const ConstPermutationView& inverseSegmentsPermutation )
{
   this->embeddedSegmentsView.bind( *(EmbeddedSegmentsView*) ( &embeddedSegmentsView ) );
   this->segmentsPermutationView.bind( const_cast< IndexType* >( segmentsPermutation.getData() ),
                                       segmentsPermutation.getSize() );
   this->inverseSegmentsPermutationView.bind( const_cast< IndexType* >( inverseSegmentsPermutation.getData() ),
                                              inverseSegmentsPermutation.getSize() );
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsView< EmbeddedSegments >::getView() -> ViewType
{
   return { this->getEmbeddedSegmentsView(), this->getSegmentsPermutationView(), this->getInverseSegmentsPermutationView() };
}

template< typename EmbeddedSegments >
__cuda_callable__
auto
SortedSegmentsView< EmbeddedSegments >::getConstView() const -> ConstViewType
{
   EmbeddedSegmentsConstView embeddedSegmentsView;
   embeddedSegmentsView.bind( this->getEmbeddedSegmentsView() );
   ConstPermutationView segmentsPermutation = this->getSegmentsPermutationView();
   ConstPermutationView inverseSegmentsPermutation = this->getInverseSegmentsPermutationView();
   ConstViewType view;
   view.bind( embeddedSegmentsView, segmentsPermutation, inverseSegmentsPermutation );
   return view;
}

}  // namespace TNL::Algorithms::Segments
