// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "MultidiagonalMatrixView.h"

namespace TNL::Matrices {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
MultidiagonalMatrixView< Real, Device, Index, Organization >::MultidiagonalMatrixView(
   typename Base::ValuesViewType values,
   typename Base::DiagonalOffsetsView diagonalOffsets,
   typename Base::HostDiagonalOffsetsView hostDiagonalOffsets,
   typename Base::IndexerType indexer )
: Base( std::move( values ), std::move( diagonalOffsets ), std::move( hostDiagonalOffsets ), std::move( indexer ) )
{}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::bind( MultidiagonalMatrixView view )
{
   Base::bind( std::move( view.values ),
               std::move( view.diagonalOffsets ),
               std::move( view.hostDiagonalOffsets ),
               std::move( view.indexer ) );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
auto
MultidiagonalMatrixView< Real, Device, Index, Organization >::getView() -> ViewType
{
   return {
      this->getValues().getView(), this->diagonalOffsets.getView(), this->hostDiagonalOffsets.getView(), this->getIndexer()
   };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
auto
MultidiagonalMatrixView< Real, Device, Index, Organization >::getConstView() const -> ConstViewType
{
   return { this->getValues.getConstView(),
            this->diagonalOffsets.getConstView(),
            this->hostDiagonalOffsets.getConstView(),
            this->getIndexer() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::save( File& file ) const
{
   file.save( &this->rows );
   file.save( &this->columns );
   file << this->values << this->diagonalOffsets;
}

}  // namespace TNL::Matrices
