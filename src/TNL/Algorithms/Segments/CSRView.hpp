// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "CSRView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index >
__cuda_callable__
CSRView< Device, Index >::CSRView( typename Base::OffsetsView offsets )
: Base( std::move( offsets ) )
{}

template< typename Device, typename Index >
__cuda_callable__
void
CSRView< Device, Index >::bind( CSRView view )
{
   Base::bind( std::move( view.offsets ) );
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

}  // namespace TNL::Algorithms::Segments
