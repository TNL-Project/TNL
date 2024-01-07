// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "BiEllpackView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
BiEllpackView< Device, Index, Organization, WarpSize >::BiEllpackView( Index size,
                                                                       Index storageSize,
                                                                       typename Base::OffsetsView segmentsPermutation,
                                                                       typename Base::OffsetsView groupPointers )
: Base( size, storageSize, std::move( segmentsPermutation ), std::move( groupPointers ) )
{}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
void
BiEllpackView< Device, Index, Organization, WarpSize >::bind( BiEllpackView view )
{
   this->size = view.size;
   this->storageSize = view.storageSize;
   this->segmentsPermutation.bind( std::move( view.segmentsPermutation ) );
   this->groupPointers.bind( std::move( view.groupPointers ) );
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
typename BiEllpackView< Device, Index, Organization, WarpSize >::ViewType
BiEllpackView< Device, Index, Organization, WarpSize >::getView()
{
   return { this->getSize(), this->getStorageSize(), this->getSegmentsPermutationView(), this->getGroupPointersView() };
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, Organization, WarpSize >::getConstView() const -> ConstViewType
{
   return { this->getSize(), this->getStorageSize(), this->getSegmentsPermutationView(), this->getGroupPointersView() };
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
void
BiEllpackView< Device, Index, Organization, WarpSize >::save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file << this->segmentsPermutation << this->groupPointers;
}

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
void
BiEllpackView< Device, Index, Organization, WarpSize >::load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file >> this->segmentsPermutation >> this->groupPointers;
}

}  // namespace TNL::Algorithms::Segments
