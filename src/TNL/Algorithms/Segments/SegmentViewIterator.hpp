// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/SegmentView.h>
#include <TNL/Assert.h>

namespace TNL::Algorithms::Segments {

template< typename SegmentView >
__cuda_callable__
SegmentViewIterator< SegmentView >::SegmentViewIterator( const SegmentViewType& segmentView, IndexType localIdx )
: segmentView( segmentView ), localIdx( localIdx )
{}

template< typename SegmentView >
__cuda_callable__
bool
SegmentViewIterator< SegmentView >::operator==( const SegmentViewIterator& other ) const
{
   return &this->segmentView == &other.segmentView && localIdx == other.localIdx;
}

template< typename SegmentView >
__cuda_callable__
bool
SegmentViewIterator< SegmentView >::operator!=( const SegmentViewIterator& other ) const
{
   return ! ( other == *this );
}

template< typename SegmentView >
__cuda_callable__
SegmentViewIterator< SegmentView >&
SegmentViewIterator< SegmentView >::operator++()
{
   if( localIdx < segmentView.getSize() )
      localIdx++;
   return *this;
}

template< typename SegmentView >
__cuda_callable__
SegmentViewIterator< SegmentView >&
SegmentViewIterator< SegmentView >::operator--()
{
   if( localIdx > 0 )
      localIdx--;
   return *this;
}

template< typename SegmentView >
__cuda_callable__
auto
SegmentViewIterator< SegmentView >::operator*() const -> SegmentElementType
{
   return { this->segmentView.getSegmentIndex(), this->localIdx, this->segmentView.getGlobalIndex( this->localIdx ) };
}

}  // namespace TNL::Algorithms::Segments
