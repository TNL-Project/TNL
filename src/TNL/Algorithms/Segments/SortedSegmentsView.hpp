// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SortedSegmentsView.h"

namespace TNL::Algorithms::Segments {

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
SortedSegmentsView< EmbeddedSegments, Index >::SortedSegmentsView( typename Base::EmbeddedSegmentsView embeddedSegmentsView,
                                                                   typename Base::PermutationView segmentsPermutation,
                                                                   typename Base::PermutationView inverseSegmentsPermutation )
{
   Base::bind( std::move( embeddedSegmentsView ), std::move( segmentsPermutation ), std::move( inverseSegmentsPermutation ) );
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
void
SortedSegmentsView< EmbeddedSegments, Index >::bind( const SortedSegmentsView& view )
{
   this->embeddedSegmentsView.bind( view.getEmbeddedSegmentsView() );
   this->segmentsPermutation.bind( view.getSegmentsPermutationView() );
   this->inverseSegmentsPermutation.bind( view.getInverseSegmentsPermutationView() );
}

template< typename EmbeddedSegments, typename Index >
void
SortedSegmentsView< EmbeddedSegments, Index >::save( File& file ) const
{
   file << this->embeddedSegmentsView << this->segmentsPermutationView << this->inverseSegmentsPermutationView;
}

template< typename EmbeddedSegments, typename Index >
void
SortedSegmentsView< EmbeddedSegments, Index >::load( File& file )
{
   file >> this->embeddedSegmentsView >> this->segmentsPermutationView >> this->inverseSegmentsPermutationView;
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsView< EmbeddedSegments, Index >::getView() -> ViewType
{
   return { this->getEmbeddedSegmentsView(), this->getSegmentsPermutationView(), this->getInverseSegmentsPermutationView() };
}

template< typename EmbeddedSegments, typename Index >
__cuda_callable__
auto
SortedSegmentsView< EmbeddedSegments, Index >::getConstView() const -> ConstViewType
{
   return ConstViewType( this->getEmbeddedSegmentsView().getConstView(),
                         this->getSegmentsPermutationView().getConstView(),
                         this->getInverseSegmentsPermutationView().getConstView() );
}

}  // namespace TNL::Algorithms::Segments
